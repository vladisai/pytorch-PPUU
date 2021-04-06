from dataclasses import dataclass
from typing import Optional
from concurrent import futures
import torch

import nevergrad as ng

from ppuu import configs

from ppuu.modeling.km import predict_states_seq, predict_states


def repeat_batch(value, times, interleave=False):
    if interleave:
        return value.repeat_interleave(times, dim=0)
    else:
        return value.repeat(times, *([1] * (len(value.shape) - 1)))


class CE:
    """Adapted from:
    https://github.com/homangab/gradcem/blob/master/mpc/cem.py
    """

    def __init__(
        self,
        batch_size=1,
        horizon=30,
        n_iter=10,
        population_size=30,
        top_size=10,
        variance=4,
        gd=True,
        lr=0.1,
        repeat_step=5,
    ):
        self.a_size = 2
        self.batch_size = batch_size
        self.horizon = horizon
        self.device = torch.device("cuda")
        self.n_iter = n_iter
        self.population_size = population_size
        self.top_size = top_size
        self.variance = variance
        self.gd = gd
        self.lr = lr
        self.repeat_step = repeat_step
        self.plan_length = int(self.horizon / self.repeat_step)

    def plan(self, get_cost):
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        a_mu = torch.zeros(self.batch_size, 1, self.plan_length, self.a_size, device=self.device)
        a_std = self.variance * torch.ones(self.batch_size, 1, self.plan_length, self.a_size, device=self.device)
        actions = (
            a_mu
            + a_std
            * torch.randn(self.batch_size, self.population_size, self.plan_length, self.a_size, device=self.device,)
        ).repeat_interleave(self.repeat_step, dim=2)
        actions.requires_grad = True

        optimizer = torch.optim.SGD((actions,), self.lr)

        for _ in range(self.n_iter):
            # now we want to get the rewards for those actions.
            costs = get_cost(actions.view(-1, self.horizon, self.a_size), keep_batch_dim=True,).view(
                self.batch_size, self.population_size
            )

            if self.gd:
                optimizer.zero_grad()
                costs.mean().backward()
                torch.nn.utils.clip_grad_norm_(actions, 1.0, norm_type="inf")
                optimizer.step()

            # get the indices of the best cost elements
            values, topk = costs.topk(self.top_size, dim=1, largest=False, sorted=True,)

            # pick the actions that correspond to the best elements
            best_actions = actions.view(self.batch_size, self.population_size, self.horizon, self.a_size,).gather(
                dim=1, index=topk.view(*topk.shape, 1, 1).repeat(1, 1, self.horizon, 2)
            )

            # Update belief with new means and standard deviations
            a_mu = best_actions.mean(dim=1, keepdim=True)
            a_std = best_actions.std(dim=1, unbiased=False, keepdim=True)

            resample_actions = a_mu + a_std * torch.randn(
                self.batch_size,
                self.population_size - self.top_size,
                self.plan_length,
                self.a_size,
                device=self.device,
            ).repeat_interleave(self.repeat_step, dim=2)

            actions.data = torch.cat([best_actions, resample_actions], dim=1)

        return actions[:, 0]


class MPCKMPolicy(torch.nn.Module):
    @dataclass
    class Config(configs.ConfigBase):
        n_iter: int = 10
        lr: float = 0.1
        unfold_len: int = 30
        timestep: float = 0.1
        update_ref_period: int = 100
        use_fm: bool = True
        optimizer: str = "Adam"
        optimizer_budget: Optional[int] = None
        lbfgs_log_barrier_alpha: float = 0.01
        fm_unfold_variance: float = 1.0
        fm_unfold_samples: int = 1
        fm_unfold_samples_agg: str = "max"
        planning_freq: int = 1
        lambda_j_mpc: float = 0.0
        batch_size: int = 1
        ce_repeat_step: int = 5
        unfolding_agg: str = "max"
        km_noise: float = 0.0
        save_opt_stats: bool = False

    OPTIMIZER_DICT = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "Nevergrad": "hhh",
        "LBFGS": torch.optim.LBFGS,
    }

    def __init__(
        self, forward_model, cost, normalizer, config, visualizer=None,
    ):
        super().__init__()

        self.cost = cost
        self.forward_model = forward_model
        self.normalizer = normalizer
        self.config = config

        self.last_actions = None
        self.visualizer = visualizer
        self.reset()

    def reset(self):
        self.last_actions = None
        self.ctr = 0

    def unfold_km(self, states, actions):
        """
        Autoregressively applies km state prediction to states with given actions.
            states shape : batch, state_dim
            actions shape : batch, unfold_len, action_dim
        Returns:
            predicted_states, shape = batch, unfold_len, state_dim

        These calculations are faster on cpu, so we transfer the data before we do it.
        """
        if self.config.km_noise == 0:
            return predict_states_seq(states, actions, self.normalizer, timestep=self.config.timestep,)
        else:
            predictions = []
            for i in range(self.config.unfold_len):
                states = predict_states(
                    states,
                    actions[:, i],
                    self.normalizer,
                    timestep=self.config.timestep,
                    noise=self.config.km_noise,
                )
                predictions.append(states)
            return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def unfold_fm(self, images, states, actions):
        """
        Autoregressively applies fm prediction to get reference images and states.
            states shape : batch, state_dim
            actions shape : batch, unfold_len, action_dim
        Returns:
            predicted_images, shape = batch, unfold_len, images channels, images h, images w
            predicted_states, shape = batch, unfold_len, state_dim
        """

        actions_per_fm_timestep = int(0.1 / self.config.timestep)
        if (
            actions_per_fm_timestep == 0
            or self.config.unfold_len % actions_per_fm_timestep != 0
            or not self.config.use_fm
        ):
            ref_states = self.unfold_km(states[..., -1, :].view(-1, 5), torch.zeros_like(actions))
            return (
                images[:, -1].repeat(1, self.config.unfold_len, 1, 1, 1),
                ref_states,
            )
        else:
            actions = actions.view(actions.shape[0], -1, actions_per_fm_timestep, 2)
            avg_actions = actions.mean(dim=2)

            # We make a batch of the same values, but we use different latents.
            # The motivation is to get multiple fm predictions and plan through that to get
            # better results that reflect the uncertainty.

            avg_actions = repeat_batch(avg_actions, self.config.fm_unfold_samples)
            images = repeat_batch(images, self.config.fm_unfold_samples)
            states = repeat_batch(states, self.config.fm_unfold_samples)

            Z = torch.randn(*avg_actions.shape[:2], 32) * self.config.fm_unfold_variance

            unfolding = self.forward_model.model.unfold(
                actions_or_policy=avg_actions,
                batch={"input_images": images.cuda(), "input_states": states.cuda(),},
                Z=Z,
            )

            if self.config.fm_unfold_samples_agg == "max":
                ref_images = (
                    unfolding["pred_images"]
                    .max(dim=0, keepdim=True)
                    .values.repeat_interleave(actions_per_fm_timestep, dim=1)
                )
            elif self.config.fm_unfold_samples_agg == "mean":
                ref_images = (
                    unfolding["pred_images"].mean(dim=0, keepdim=True).repeat_interleave(actions_per_fm_timestep, dim=1)
                )
            elif self.config.fm_unfold_samples_agg == "keep":
                ref_images = unfolding["pred_images"].repeat_interleave(actions_per_fm_timestep, dim=1)

            if self.config.fm_unfold_samples_agg == "keep":
                ref_states = unfolding["pred_states"].repeat_interleave(actions_per_fm_timestep, dim=1)
            else:
                # TODO: this has to also account for the fact that other cars are probably moving at the same rate as us.
                ref_states = unfolding["pred_states"][:1].repeat_interleave(actions_per_fm_timestep, dim=1)

            return ref_images, ref_states

    def __call__(
        self, images, states, normalize_inputs=False, normalize_outputs=False, car_size=None, init=None, metadata=None,
        gt_future=None,
    ):
        device = states.device

        if self.ctr % self.config.planning_freq > 0:
            actions = self.last_actions[:, self.ctr % self.config.planning_freq]

            if normalize_outputs:
                actions = self.normalizer.unnormalize_actions(actions.data)

            print("final actions for", self.ctr, "are", actions)

            self.ctr += 1

            return actions.detach()

        if gt_future is not None:
            gt_future_values = gt_future() # gt future is a lambda. this is to save time when we don't plan every step.
            if gt_future_values is not None:
                ref_images = self.normalizer.normalize_images(gt_future_values.images.unsqueeze(0)).to(device)
                ref_states = self.normalizer.normalize_states(gt_future_values.states.unsqueeze(0)).to(device)
        else:
            gt_future_values = None

        # if self.ctr == 99:
        #     dump_dict = dict(images=images, states=states, car_size=car_size,)
        #     torch.save(dump_dict, "bad_example.dump")

        if self.visualizer:
            self.visualizer.step_reset()

        if normalize_inputs:
            states = self.normalizer.normalize_states(states.clone())
            images = self.normalizer.normalize_images(images)
            car_size = torch.tensor(car_size).unsqueeze(0)

        full_states = states.unsqueeze(0)
        full_images = images[:, :3].unsqueeze(0)
        states = states[..., -1, :].view(-1, 5)
        images = images[..., -1, :, :, :].view(-1, 1, 4, 117, 24)
        # if self.last_actions is not None:
        #     actions = torch.cat(
        #         (self.last_actions[:, 1:-1], self.last_actions[:, -2].unsqueeze(1).repeat(1, 2, 1)), dim=1
        #     )
        #     actions = torch.tensor(actions, requires_grad=True)
        # else:

        # Zero normalized maps to slight acceleration when normalized. We make sure we start from
        # true zero.
        actions = torch.cat(
            [
                # The first action is always just zeros.
                torch.zeros(1, self.config.unfold_len, 2, device=device),
                # The rest are random repeated.
                7 * torch.randn(self.config.batch_size - 1, 1, 2, device=device).repeat(1, self.config.unfold_len, 1),
            ],
            dim=0,
        )
        # actions[:, :, 0] = 30
        # best_actions = torch.zeros_like(actions[:1])
        best_actions = actions[:1]

        actions[:, :, 1] = 0
        best_cost = 1e10

        if init is not None:
            actions[0, 0] = init

        actions = self.normalizer.normalize_actions(actions)
        self.cost.traj_landscape = False
        actions.requires_grad = True

        def get_cost(actions, keep_batch_dim=False, unfolding_agg="max"):
            batch_size = actions.shape[0]
            unfoldings_size = ref_images.shape[0]

            rep_states = repeat_batch(states, batch_size)
            rep_actions = repeat_batch(actions, unfoldings_size)

            pred_states = repeat_batch(self.unfold_km(rep_states, actions), unfoldings_size)

            inputs = {
                "input_images": repeat_batch(images, batch_size * unfoldings_size),
                "input_states": repeat_batch(states.unsqueeze(1), batch_size * unfoldings_size),
                "car_sizes": repeat_batch(car_size, batch_size * unfoldings_size),
                "ref_states": repeat_batch(ref_states, batch_size, interleave=True),
                "ref_images": repeat_batch(ref_images, batch_size, interleave=True),
            }
            predictions = {
                "pred_states": pred_states,
                "pred_images": repeat_batch(ref_images, batch_size, interleave=True),
                "pred_actions": repeat_batch(actions, unfoldings_size),
            }

            jerk_actions = actions
            if self.last_actions is not None:
                # Cat the last action so we account for what action we performed in the past when calculating jerk.
                jerk_actions = torch.cat([repeat_batch(self.last_actions[:, :1], batch_size), actions,], dim=1,)

            gamma_mask = (
                torch.tensor([self.cost.config.gamma ** t for t in range(jerk_actions.shape[1] - 1)])
                .cuda()
                .unsqueeze(0)
            )
            loss_j = (jerk_actions[:, 1:] - jerk_actions[:, :-1]).norm(2, 2).pow(2).mul(gamma_mask).mean(dim=1)

            # costs = self.cost.compute_state_costs_for_training(inputs, pred_images, pred_states, actions, car_size)
            costs = self.cost.calculate_cost(inputs, predictions)

            result = costs["policy_loss"] + repeat_batch(loss_j, unfoldings_size) * self.config.lambda_j_mpc

            # TODO: double check this is correct.
            result = result.view(unfoldings_size, batch_size)

            if unfolding_agg == "mean":
                result = result.mean(dim=0)
            elif unfolding_agg == "max":
                result = result.max(dim=0).values

            if keep_batch_dim:
                return result
            else:
                return result.mean()

        if self.config.optimizer in ["SGD", "Adam"]:
            optimizer = self.OPTIMIZER_DICT[self.config.optimizer]((actions,), self.config.lr)

            for i in range(self.config.n_iter):
                if i % self.config.update_ref_period == 0 and gt_future_values is None:
                    # We don't regenerate this if gt is passed!
                    ref_images, ref_states = self.unfold_fm(full_images, full_states, best_actions)

                # costs = self.cost.compute_state_costs_for_training(inputs, pred_images, pred_states, actions, car_size)
                optimizer.zero_grad()
                if i == self.config.n_iter - 1 and self.visualizer is not None:
                    self.cost.traj_landscape = True

                cost = get_cost(actions, keep_batch_dim=True, unfolding_agg=self.config.unfolding_agg)

                if i == self.config.n_iter - 1:
                    self.cost.traj_landscape = False

                cost.mean().backward()
                a_grad = actions.grad[0, 0].clone() # save for plotting later
                if not torch.isnan(actions.grad).any():
                    # this is definitely needed, judging from some examples I saw where gradient is 50
                    torch.nn.utils.clip_grad_norm_(actions, 1.0, norm_type="inf")
                    optimizer.step()
                else:
                    print('NaN grad!')

                values, indices = cost.min(dim=0)
                # if cost[indices] < best_cost:
                #     best_actions = actions[indices].unsqueeze(0).clone()
                #     best_cost = cost[indices]
                best_cost = cost[indices]
                best_actions = actions[indices].unsqueeze(0).clone()

                if self.visualizer:
                    unnormalized_actions = self.normalizer.unnormalize_actions(actions.data)
                    self.visualizer.update_values(
                        best_cost.item(),
                        unnormalized_actions[0, 0, 0].item(),
                        unnormalized_actions[0, 0, 1].item(),
                        a_grad[0].item(),
                        a_grad[1].item(),
                    )

        elif self.config.optimizer == "Nevergrad":
            i = 0

            def ng_get_cost(actions):
                if i == 0 and self.visualizer is not None:
                    self.cost.traj_landscape = True
                i += 1

                a = torch.tensor(actions).float().cuda()
                cost = get_cost(a)

                if metadata is not None and "cost" not in metadata:
                    metadata["cost"] = cost

                if i == 0:
                    self.cost.traj_landscape = False

                if self.visualizer:
                    unnormalized_actions = self.normalizer.unnormalize_actions(a.data)
                    self.visualizer.update_values(
                        cost.item(), unnormalized_actions[0, 0, 0].item(), unnormalized_actions[0, 0, 1].item(),
                    )

                return cost.item()

            ref_images, ref_states = self.unfold_fm(full_images, full_states, best_actions)

            parametrization = ng.p.Array(shape=actions.shape)
            parametrization.set_bounds(-5, 5)
            optim = ng.optimizers.registry["CMA"](parametrization=parametrization, budget=self.config.optimizer_budget,)
            with futures.ThreadPoolExecutor(
                max_workers=optim.num_workers
            ) as executor:  # the executor will evaluate the function in multiple threads
                recommendation = optim.minimize(ng_get_cost, executor=executor)
                actions = torch.tensor(recommendation.value).float().cuda()

        elif self.config.optimizer == "LBFGS":
            optimizer = self.OPTIMIZER_DICT[self.config.optimizer](
                (actions,), lr=self.config.lr, max_iter=self.config.n_iter
            )
            if gt_future_values is None:
                # We don't regenerate this if gt is passed!
                ref_images, ref_states = self.unfold_fm(full_images, full_states, best_actions)

            if self.visualizer is not None:
                self.cost.traj_landscape = True

            def lbfgs_closure():
                optimizer.zero_grad()
                # We want to add log-barrier function to not let LBFGS go beyond a reasonable interval for optimization.
                cost = get_cost(actions, keep_batch_dim=True, unfolding_agg=self.config.unfolding_agg) - self.config.lbfgs_log_barrier_alpha * (1.0 - actions.abs()).log().sum()

                if self.visualizer:
                    unnormalized_actions = self.normalizer.unnormalize_actions(actions.data)
                    self.visualizer.update_values(
                        cost.mean().item(), unnormalized_actions[0, 0, 0].item(), unnormalized_actions[0, 0, 1].item(),
                    )

                self.cost.traj_landscape = False

                cmean = cost.mean()
                cmean.backward()

                return cmean

            optimizer.step(lbfgs_closure)

            if self.visualizer:
                unnormalized_actions = self.normalizer.unnormalize_actions(actions.data)
                self.visualizer.update_values(
                    0.0, unnormalized_actions[0, 0, 0].item(), unnormalized_actions[0, 0, 1].item(),
                )

            best_actions = actions[:1]

        elif self.config.optimizer == "CE":
            if gt_future is None:
                ref_images, ref_states = self.unfold_fm(full_images, full_states, best_actions)
            ce = CE(
                batch_size=1,
                horizon=self.config.unfold_len,
                n_iter=self.config.n_iter,
                repeat_step=self.config.ce_repeat_step,
            )
            best_actions = ce.plan(get_cost)

            if self.visualizer:
                unnormalized_actions = self.normalizer.unnormalize_actions(best_actions.data)
                self.visualizer.update_values(
                    0.0, unnormalized_actions[0, 0, 0].item(), unnormalized_actions[0, 0, 1].item(),
                )

        self.cost.traj_landscape = False

        if self.visualizer is not None:
            self.visualizer.update_plot()

        self.last_actions = best_actions

        actions = best_actions[:, 0]

        if normalize_outputs:
            actions = self.normalizer.unnormalize_actions(actions.data)

        print("final actions for", self.ctr, "are", actions)

        self.ctr += 1

        return actions.detach()


class MPCFMPolicy(torch.nn.Module):
    @dataclass
    class Config(configs.ConfigBase):
        n_iter: int = 10
        lr: float = 0.1
        unfold_len: int = 30
        timestep: float = 0.1
        update_ref_period: int = 100
        use_fm: bool = True
        optimizer: str = "Adam"
        fm_unfold_variance: float = 1.0
        fm_unfold_samples: int = 1
        fm_unfold_samples_agg: str = "max"
        planning_freq: int = 1
        lambda_j_mpc: float = 0.0
        batch_size: int = 1
        ce_repeat_step: int = 5
        unfolding_agg: str = "max"
        save_opt_stats: bool = False

    OPTIMIZER_DICT = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
    }

    def __init__(
        self, forward_model, cost, normalizer, config, visualizer=None,
    ):
        super().__init__()

        self.cost = cost
        self.forward_model = forward_model
        self.normalizer = normalizer
        self.config = config

        self.last_actions = None
        self.visualizer = visualizer
        self.reset()

    def reset(self):
        self.last_actions = None
        self.ctr = 0
        self.optimizer_stats = None

    def unfold_fm(self, images, states, actions, Z=None):
        """
        Autoregressively applies fm prediction to get reference images and states.
            states shape : batch, state_dim
            actions shape : batch, unfold_len, action_dim
            Z shape : batch, unfold_len, 32
        Returns:
            predicted_images, shape = batch, unfold_len, images channels, images h, images w
            predicted_states, shape = batch, unfold_len, state_dim
        """

        actions_per_fm_timestep = int(0.1 / self.config.timestep)
        if (
            actions_per_fm_timestep == 0
            or self.config.unfold_len % actions_per_fm_timestep != 0
            or not self.config.use_fm
        ):
            ref_states = self.unfold_km(states[..., -1, :].view(-1, 5), torch.zeros_like(actions))
            return (
                images[:, -1].repeat(1, self.config.unfold_len, 1, 1, 1),
                ref_states,
            )
        else:
            # We make a batch of the same values, but we use different latents.
            # The motivation is to get multiple fm predictions and plan through that to get
            # better results that reflect the uncertainty.

            actions = repeat_batch(actions, self.config.fm_unfold_samples)
            images = repeat_batch(images, self.config.fm_unfold_samples)
            states = repeat_batch(states, self.config.fm_unfold_samples)

            if Z is None:
                Z = torch.randn(*actions.shape[:2], 32) * self.config.fm_unfold_variance
                print('WARN: reinitializing Z')
            # else:
            #     print('using the provided z', Z.view(self.config.fm_unfold_samples, self.config.batch_size, -1).sum(dim=-1))

            unfolding = self.forward_model.model.unfold(
                actions_or_policy=actions,
                batch={"input_images": images.cuda(), "input_states": states.cuda(),},
                Z=Z,
            )

            if self.config.fm_unfold_samples_agg == "max":
                ref_images = (
                    unfolding["pred_images"]
                    .max(dim=0, keepdim=True)
                    .values
                )
            elif self.config.fm_unfold_samples_agg == "mean":
                ref_images = (
                    unfolding["pred_images"].mean(dim=0, keepdim=True)
                )
            elif self.config.fm_unfold_samples_agg == "keep":
                ref_images = unfolding["pred_images"]

            if self.config.fm_unfold_samples_agg == "keep":
                ref_states = unfolding["pred_states"]
            else:
                # TODO: this has to also account for the fact that other cars are probably moving at the same rate as us.
                ref_states = unfolding["pred_states"][:1]

            return ref_images, ref_states, unfolding["Z"]

    def __call__(
        self, images, states, normalize_inputs=False, normalize_outputs=False, car_size=None, init=None, metadata=None,
    ):
        device = states.device

        if self.ctr % self.config.planning_freq > 0:
            actions = self.last_actions[:, self.ctr % self.config.planning_freq]

            if normalize_outputs:
                actions = self.normalizer.unnormalize_actions(actions.data)

            print("final actions for", self.ctr, "are", actions)

            self.ctr += 1

            return actions.detach()

        # if self.ctr == 99:
        #     dump_dict = dict(images=images, states=states, car_size=car_size,)
        #     torch.save(dump_dict, "bad_example.dump")

        if self.visualizer:
            self.visualizer.step_reset()

        if normalize_inputs:
            states = self.normalizer.normalize_states(states.clone())
            images = self.normalizer.normalize_images(images)
            car_size = torch.tensor(car_size).unsqueeze(0)

        full_states = states.unsqueeze(0)
        full_images = images[:, :3].unsqueeze(0)
        states = states[..., -1, :].view(-1, 5)
        images = images[..., -1, :, :, :].view(-1, 1, 4, 117, 24)
        # if self.last_actions is not None:
        #     actions = torch.cat(
        #         (self.last_actions[:, 1:-1], self.last_actions[:, -2].unsqueeze(1).repeat(1, 2, 1)), dim=1
        #     )
        #     actions = torch.tensor(actions, requires_grad=True)
        # else:

        # Zero normalized maps to slight acceleration when normalized. We make sure we start from
        # true zero.
        actions = torch.cat(
            [
                # The first action is always just zeros.
                torch.zeros(1, self.config.unfold_len, 2, device=device),
                # The rest are random repeated.
                torch.randn(self.config.batch_size - 1, 1, 2, device=device).repeat(1, self.config.unfold_len, 1),
            ],
            dim=0,
        )
        # actions[:, :, 0] = 30
        # best_actions = torch.zeros_like(actions[:1])
        actions = torch.randn_like(actions) * 0.01
        actions[:, :, 1] = 0
        best_actions = actions[:1]

        # TODO: remove the changes to init in actions
        best_cost = 1e10

        if init is not None:
            actions[0, 0] = init

        actions = self.normalizer.normalize_actions(actions)
        self.cost.traj_landscape = False
        actions.requires_grad = True

        def get_cost(actions, Z=None, keep_batch_dim=False, unfolding_agg="max", metadata=None):
            batch_size = actions.shape[0]
            unfoldings_size = self.config.fm_unfold_samples

            rep_states = repeat_batch(full_states, batch_size)
            rep_images = repeat_batch(full_images, batch_size)
            rep_actions = repeat_batch(actions, unfoldings_size)

            pred_images, pred_states, Z = self.unfold_fm(rep_images, rep_states, rep_actions, Z)

            if metadata is not None:
                metadata['pred_images'].append(pred_images.clone().detach())
                metadata['pred_states'].append(pred_states.clone().detach())

            inputs = {
                "input_images": repeat_batch(full_images, batch_size * unfoldings_size),
                "input_states": repeat_batch(full_states, batch_size * unfoldings_size),
                "car_sizes": repeat_batch(car_size, batch_size * unfoldings_size),
            }
            predictions = {
                "pred_states": pred_states,
                "pred_images": pred_images,
                "pred_actions": repeat_batch(actions, unfoldings_size),
                "Z": Z,
            }

            jerk_actions = actions
            if self.last_actions is not None:
                # Cat the last action so we account for what action we performed in the past when calculating jerk.
                jerk_actions = torch.cat([repeat_batch(self.last_actions[:, :1], batch_size), actions,], dim=1,)

            gamma_mask = (
                torch.tensor([self.cost.config.gamma ** t for t in range(jerk_actions.shape[1] - 1)])
                .cuda()
                .unsqueeze(0)
            )
            loss_j = (jerk_actions[:, 1:] - jerk_actions[:, :-1]).norm(2, 2).pow(2).mul(gamma_mask).mean(dim=1)

            # costs = self.cost.compute_state_costs_for_training(inputs, pred_images, pred_states, actions, car_size)
            costs = self.cost.calculate_cost(inputs, predictions)

            if metadata is not None:
                metadata['costs'].append(costs)
                metadata['predictions'].append(predictions)

            result = costs["policy_loss"] + repeat_batch(loss_j, unfoldings_size) * self.config.lambda_j_mpc

            # TODO: double check this is correct.
            result = result.view(unfoldings_size, batch_size)

            if unfolding_agg == "mean":
                result = result.mean(dim=0)
            elif unfolding_agg == "max":
                result = result.max(dim=0).values

            if keep_batch_dim:
                return result
            else:
                return result.mean()

        if self.config.optimizer in ["SGD", "Adam"]:
            optimizer = self.OPTIMIZER_DICT[self.config.optimizer]((actions,), self.config.lr)

            if (self.optimizer_stats is not None) and self.config.save_opt_stats:
                print('loading opt stats')
                optimizer.load_state_dict(self.optimizer_stats)

            # We keep the latent constant to make optimization easier.
            Z = torch.randn(self.config.batch_size, self.config.unfold_len, 32) * self.config.fm_unfold_variance
            Z = repeat_batch(Z, self.config.fm_unfold_samples, interleave=True)

            for i in range(self.config.n_iter):
                # costs = self.cost.compute_state_costs_for_training(inputs, pred_images, pred_states, actions, car_size)
                optimizer.zero_grad()
                if i == self.config.n_iter - 1 and self.visualizer is not None:
                    self.cost.traj_landscape = True

                cost = get_cost(actions, Z=Z, keep_batch_dim=True, unfolding_agg=self.config.unfolding_agg, metadata=metadata)

                if i == self.config.n_iter - 1:
                    self.cost.traj_landscape = False

                cost.mean().backward()
                a_grad = actions.grad[0, 0].clone() # save for plotting later

                if not torch.isnan(actions.grad).any():
                    # this is definitely needed, judging from some examples I saw where gradient is 50
                    torch.nn.utils.clip_grad_norm_(actions, 1.0, norm_type="inf")
                    optimizer.step()
                else:
                    print('NaN grad!')

                values, indices = cost.min(dim=0)
                # if cost[indices] < best_cost:
                #     best_actions = actions[indices].unsqueeze(0).clone()
                #     best_cost = cost[indices]
                best_cost = cost[indices]
                best_actions = actions[indices].unsqueeze(0).clone()

                if self.visualizer:
                    unnormalized_actions = self.normalizer.unnormalize_actions(actions.data)
                    self.visualizer.update_values(
                        best_cost.item(),
                        unnormalized_actions[0, 0, 0].item(),
                        unnormalized_actions[0, 0, 1].item(),
                        a_grad[0].item(),
                        a_grad[1].item(),
                    )

                self.optimizer_stats = optimizer.state_dict()

        self.cost.traj_landscape = False

        if self.visualizer is not None:
            self.visualizer.update_plot()

        self.last_actions = best_actions

        actions = best_actions[:, 0]

        if normalize_outputs:
            actions = self.normalizer.unnormalize_actions(actions.data)

        print("final actions for", self.ctr, "are", actions)

        self.ctr += 1

        return actions.detach()

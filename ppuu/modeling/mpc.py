from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Union

import torch

from ppuu import configs
from ppuu.modeling.km import predict_states, predict_states_seq

from ppuu.data.entities import StateSequence


def repeat_batch(value, times, dim=0, interleave=False):
    if interleave:
        return value.repeat_interleave(times, dim=dim)
    else:
        repeat_arr = [1] * len(value.shape)
        repeat_arr[dim] = times
        return value.repeat(*repeat_arr)


def flatten_dims(x: Union[torch.Tensor, StateSequence], flatten_dims: int):
    if torch.is_tensor(x):
        return x.contiguous().view(-1, *x.shape[flatten_dims:])
    else:
        return x.map(
            lambda y: y.contiguous().view(-1, *y.shape[flatten_dims:])
        )


def unflatten_dims(
    x: Union[torch.Tensor, StateSequence], unflatten_dims: Tuple[int, ...]
):
    if torch.is_tensor(x):
        return x.contiguous().view(*unflatten_dims, *x.shape[1:])
    else:
        return x.map(
            lambda y: y.contiguous().view(*unflatten_dims, *y.shape[1:])
        )


def flatten_2_dims(x):
    return flatten_dims(x, 2)


def flatten_f_unflatten(f, args, flatten_dims=2):
    """Flattens first flattenn_dims in x, applies
    x to the tensor, and unflattens the result to have the
    original first two dims"""
    original_shapes = [x.shape for x in args]
    args_flat = [flatten_dims(x) for x in args]
    f_x_flat = f(*args_flat)
    return unflatten_dims(f_x_flat, original_shapes[0][:flatten_dims])


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
        a_mu = torch.zeros(
            self.batch_size,
            1,
            self.plan_length,
            self.a_size,
            device=self.device,
        )
        a_std = self.variance * torch.ones(
            self.batch_size,
            1,
            self.plan_length,
            self.a_size,
            device=self.device,
        )
        actions = (
            a_mu
            + a_std
            * torch.randn(
                self.batch_size,
                self.population_size,
                self.plan_length,
                self.a_size,
                device=self.device,
            )
        ).repeat_interleave(self.repeat_step, dim=2)
        actions.requires_grad = True

        optimizer = torch.optim.SGD((actions,), self.lr)

        for _ in range(self.n_iter):
            # now we want to get the rewards for those actions.
            costs = get_cost(
                actions.view(-1, self.horizon, self.a_size),
                keep_batch_dim=True,
            ).view(self.batch_size, self.population_size)

            if self.gd:
                optimizer.zero_grad()
                costs.mean().backward()
                torch.nn.utils.clip_grad_norm_(actions, 1.0, norm_type="inf")
                optimizer.step()

            # get the indices of the best cost elements
            values, topk = costs.topk(
                self.top_size,
                dim=1,
                largest=False,
                sorted=True,
            )

            # pick the actions that correspond to the best elements
            best_actions = actions.view(
                self.batch_size,
                self.population_size,
                self.horizon,
                self.a_size,
            ).gather(
                dim=1,
                index=topk.view(*topk.shape, 1, 1).repeat(
                    1, 1, self.horizon, 2
                ),
            )

            # Update belief with new means and standard deviations
            a_mu = best_actions.mean(dim=1, keepdim=True)
            a_std = best_actions.std(dim=1, unbiased=False, keepdim=True)

            resample_actions = (
                a_mu
                + a_std
                * torch.randn(
                    self.batch_size,
                    self.population_size - self.top_size,
                    self.plan_length,
                    self.a_size,
                    device=self.device,
                ).repeat_interleave(self.repeat_step, dim=2)
            )

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
        self,
        forward_model,
        cost,
        normalizer,
        config,
        visualizer=None,
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
        """A function to be called before each new episode"""
        self.last_actions = None
        self.ctr = 0
        self.fm_fallback = False

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
            return predict_states_seq(
                states,
                actions,
                self.normalizer,
                timestep=self.config.timestep,
            )
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
    def unfold_fm(
        self, conditional_state_seq: StateSequence, actions: torch.Tensor
    ) -> StateSequence:
        """
        Autoregressively applies fm prediction to get reference images and states.
            states shape : batch, state_dim
            actions shape : batch, unfold_len, action_dim
        Returns:
            predicted_images, shape = batch, unfold_len, images channels, images h, images w
            predicted_states, shape = batch, unfold_len, state_dim
        """

        print("Running fm")

        # We make a batch of the same values, but we use different latents.
        # The motivation is to get multiple fm predictions and plan through that to get
        # better results that reflect the uncertainty.

        # we repeat the conditional frames to generate multiple unfold samples.
        rep_conditional_state_seq = conditional_state_seq.map(
            lambda x: repeat_batch(
                x.unsqueeze(1), self.config.fm_unfold_samples, dim=1
            )
        )

        actions = actions.unsqueeze(1).repeat_interleave(
            self.config.fm_unfold_samples, dim=1
        )
        # todo checkout what the hell is going on with dimensions when we want batched predictions

        Z = (
            torch.randn(*actions.shape[:2], 32)
            * self.config.fm_unfold_variance
        ).to(actions.device)

        print(
            f"inside unfold_fm {Z.shape=}, {actions.shape=}, {rep_conditional_state_seq.images.shape=}"
        )

        unfolding = flatten_f_unflatten(
            self.forward_model.unfold,
            (rep_conditional_state_seq, actions, Z),
            flatten_dims=2,
        )

        if self.config.fm_unfold_samples_agg == "max":
            images = unfolding.images.max(dim=0, keepdim=True).values
        elif self.config.fm_unfold_samples_agg == "mean":
            images = unfolding.images.mean(dim=0, keepdim=True)
        elif self.config.fm_unfold_samples_agg == "keep":
            images = unfolding.images

        if self.config.fm_unfold_samples_agg == "keep":
            result_seq = StateSequence(
                images=images,
                states=unfolding.states,
                car_size=unfolding.car_size,
                ego_car_image=unfolding.ego_car_image,
            )

        else:
            states = unfolding["pred_states"][:1]
            result_seq = StateSequence(
                images=images,
                states=unfolding.states[:1],
                car_size=unfolding.car_size[:1],
                ego_car_image=unfolding.ego_car_image[:1],
            )

        return result_seq

    def get_cost_batched(
        self,
        input_images,
        input_states,
        car_size,
        pred_images,
        pred_states,
        actions,
        keep_batch_dim=False,
    ):
        inputs = {
            "input_images": input_images,
            "input_states": input_states,
            "car_sizes": car_size,
            "ref_states": pred_states,
            "ref_images": pred_images,
        }

        predictions = {
            "pred_states": pred_states,
            "pred_images": pred_images,
            "pred_actions": actions,
        }
        costs = self.cost.calculate_cost(inputs, predictions)
        return costs["policy_loss"].mean()

    def batched(
        self,
        images,
        states,
        pred_images,
        pred_states,
        normalize_inputs=False,
        normalize_outputs=False,
        car_size=None,
        init=None,
        metadata=None,
    ):
        """
        This function is used for target prop.
        """

        if normalize_inputs:
            states = self.normalizer.normalize_states(states.clone())
            images = self.normalizer.normalize_images(images)
            car_size = torch.tensor(car_size).unsqueeze(0)

        states = states[..., -1, :].view(-1, 5)
        images = images[..., -1, :, :, :].view(-1, 1, 4, 117, 24)

        actions = init.detach().clone()
        actions.requires_grad = True

        self.cost.traj_landscape = False

        if metadata is not None:
            metadata["costs"] = []

        if self.config.optimizer in ["SGD", "Adam"]:
            optimizer = self.OPTIMIZER_DICT[self.config.optimizer](
                (actions,), self.config.lr
            )
            for i in range(self.config.n_iter):
                optimizer.zero_grad()

                cost = self.get_cost_batched(
                    images.detach(),
                    states.detach(),
                    car_size,
                    pred_images.detach(),
                    pred_states.detach(),
                    actions,
                    keep_batch_dim=True,
                )

                cost.mean().backward()
                if metadata is not None:
                    metadata["costs"].append(cost.mean())
                if not torch.isnan(actions.grad).any():
                    # this is definitely needed, judging from some examples I saw where gradient is 50
                    torch.nn.utils.clip_grad_norm_(
                        actions, 1.0, norm_type="inf"
                    )
                    optimizer.step()
                else:
                    print("NaN grad!")

        else:
            raise NotImplementedError(
                f"{self.config.optimizer} optimizer is not supported"
            )

        return actions

    def should_replan(self) -> bool:
        """ Returns true if according to config this step has to be replanned"""
        return self.ctr % self.config.planning_freq == 0

    def _jerk_loss_with_last_action(
        self, actions: torch.Tensor
    ) -> torch.Tensor:
        actions_batch_size = actions.shape[1]
        jerk_actions = actions
        if self.last_actions is not None:
            # Cat the last action so we account for what action we performed in the past when calculating jerk.
            jerk_actions = torch.cat(
                [
                    repeat_batch(self.last_actions[:, :1], actions_batch_size),
                    actions,
                ],
                dim=1,
            )

        gamma_mask = (
            torch.tensor(
                [
                    self.cost.config.gamma ** t
                    for t in range(jerk_actions.shape[1] - 1)
                ]
            )
            .to(actions.device)
            .unsqueeze(0)
        )
        loss_j = (
            (jerk_actions[:, 1:] - jerk_actions[:, :-1])
            .norm(2, 2)
            .pow(2)
            .mul(gamma_mask)
            .mean(dim=1)
        )
        return loss_j

    def _input_context_cross_product(
        self,
        conditional_state_seq: StateSequence,
        scalar_states: torch.Tensor,
        actions: torch.Tensor,
        context_state_seq: StateSequence,
    ) -> Tuple[StateSequence, torch.Tensor, torch.Tensor, StateSequence]:
        # we repeat all inputs, and repeat interleave all contexts to achieve
        # cross-product.
        action_batch_size = actions.shape[1]
        unfoldings_batch_size = context_state_seq.images.shape[1]
        print(
            f"inside cross product {action_batch_size=}, {unfoldings_batch_size=}"
        )

        cp_conditional_state_seq = conditional_state_seq.map(
            lambda x: x.unsqueeze(1).expand(
                x.shape[0],
                action_batch_size * unfoldings_batch_size,
                *x.shape[1:],
            )
        )
        cp_scalar_states = repeat_batch(
            scalar_states, unfoldings_batch_size, dim=1
        )
        cp_actions = repeat_batch(actions, unfoldings_batch_size, dim=1)

        cp_context_state_seq = context_state_seq.map(
            lambda x: repeat_batch(
                x,
                times=action_batch_size,
                dim=1,
            )
        )
        return (
            cp_conditional_state_seq,
            cp_scalar_states,
            cp_actions,
            cp_context_state_seq,
        )

    def get_cost(
        self,
        conditional_state_seq: StateSequence,
        actions: torch.Tensor,
        context_state_seq: StateSequence,
        unfolding_agg: str = "max",
    ):
        """Gets the cost of actions.
        actions shape is batch_size, actions_batch_size, npred, 2
        context_state_seq shape is batch_size, unfoldings_size, npred, ...
        the output shape is batch_size, action_batch_size
        """
        batch_size = actions.shape[0]
        action_batch_size = actions.shape[1]
        unfolding_size = context_state_seq.images.shape[1]

        # All conditional states are of shape
        # batch, npred, ...
        rep_conditional_states = repeat_batch(
            conditional_state_seq.states.unsqueeze(1), action_batch_size, dim=1
        )

        # now we have actions and states of shape batch, actions_batch_size, ...
        pred_states = flatten_f_unflatten(
            self.unfold_km,
            (rep_conditional_states[..., -1, :], actions),
            flatten_dims=2,
        )

        # we construct repeated conditional frames
        # the rep shape is
        # batch_size, actions_batch_size ...
        rep_conditional_state_seq = conditional_state_seq.map(
            lambda x: x.unsqueeze(1).expand(
                x.shape[0],
                action_batch_size,
                *x.shape[1:],
            )
        )

        # Now we need to do a cross product between the tuple
        # (actions, pred_states) and different context images.

        # Now we'd have batch, action_batch_size * unfolding_batch_size, ...
        # shapes.
        (
            cross_product_conditional_state_seq,
            cross_product_scalar_states,
            cross_product_actions,
            cross_product_context_state_seq,
        ) = self._input_context_cross_product(
            rep_conditional_state_seq,
            pred_states,
            actions,
            context_state_seq,
        )

        # To calculate the cost we flatten the values to have
        # batch * action_batch_size * unfolding_batch_size, ... shape
        cross_product_conditional_state_seq = (
            cross_product_conditional_state_seq.map(flatten_2_dims)
        )
        cross_product_scalar_states = flatten_2_dims(
            cross_product_scalar_states
        )
        cross_product_actions = flatten_2_dims(cross_product_actions)
        cross_product_context_state_seq = cross_product_context_state_seq.map(
            flatten_2_dims
        )

        # costs = self.cost.compute_state_costs_for_training(inputs, pred_images, pred_states, actions, car_size)
        costs = self.cost.calculate_cost(
            cross_product_conditional_state_seq,
            cross_product_scalar_states,
            cross_product_actions,
            cross_product_context_state_seq,
        )

        assert list(costs.total.shape) == [
            batch_size * action_batch_size * unfolding_size
        ]

        total_cost = costs.total.view(
            batch_size, unfolding_size, action_batch_size
        )

        loss_j = (
            self._jerk_loss_with_last_action(flatten_2_dims(actions))
            .view(batch_size, action_batch_size)
            .unsqueeze(1)
        )
        result = total_cost + loss_j * self.config.lambda_j_mpc

        if unfolding_agg == "mean":
            result = result.mean(dim=1)
        elif unfolding_agg == "max":
            result = result.max(dim=1).values

        assert list(result.shape) == [batch_size, action_batch_size]

        return result

    def _init_actions(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Initializes actions."""
        actions = torch.cat(
            [
                # The first action is always just zeros.
                torch.zeros(
                    batch_size, 1, self.config.unfold_len, 2, device=device
                ),
                # The rest are random repeated.
                2
                * torch.randn(
                    batch_size, self.config.batch_size - 1, 1, 2, device=device
                ).repeat(1, 1, self.config.unfold_len, 1),
            ],
            dim=1,
        )
        actions[:, :, :, 1] = 0
        return actions

    def _get_context(
        self,
        conditional_state_seq: StateSequence,
        best_actions: torch.Tensor,
        i: int,
        current_context_state_seq: Optional[StateSequence],
        gt_future_state_seq: Optional[StateSequence],
    ):
        if gt_future_state_seq is not None:
            return gt_future_state_seq
        else:
            if i % self.config.update_ref_period != 0:
                return current_context_state_seq
            else:
                return self.unfold_fm(conditional_state_seq, best_actions)

    def plan(
        self,
        conditional_state_seq: StateSequence,
        gt_future_state_seq: Optional[StateSequence],
        metadata: Optional[dict] = None,
    ):
        if self.visualizer:
            self.visualizer.step_reset()

        # if self.last_actions is not None:
        #     actions = torch.cat(
        #         (self.last_actions[:, 1:-1], self.last_actions[:, -2].unsqueeze(1).repeat(1, 2, 1)), dim=1
        #     )
        #     actions = torch.tensor(actions, requires_grad=True)
        # else:

        # Zero normalized maps to slight acceleration when normalized. We make sure we start from
        # true zero.
        # actions[:, :, 0] = 30
        # best_actions = torch.zeros_like(actions[:1])

        actions = self._init_actions(
            batch_size=conditional_state_seq.states.shape[0],
            device=conditional_state_seq.states.device,
        )
        actions = self.normalizer.normalize_actions(actions)
        actions.requires_grad = True

        best_actions = actions[:, 0]  # we take the first action initially
        best_cost = 1e10

        print(f"{best_actions.shape=}")

        if metadata is not None:
            metadata["action_history"] = []

        self.cost.traj_landscape = False

        optimizer = self.OPTIMIZER_DICT[self.config.optimizer](
            (actions,), self.config.lr
        )

        future_context_state_seq = None

        for i in range(self.config.n_iter):
            future_context_state_seq = self._get_context(
                conditional_state_seq,
                best_actions,
                i,
                future_context_state_seq,
                gt_future_state_seq,
            )

            optimizer.zero_grad()

            cost = self.get_cost(
                conditional_state_seq,
                actions,
                context_state_seq=future_context_state_seq,
                unfolding_agg=self.config.unfolding_agg,
            )
            cost.mean().backward()

            a_grad = actions.grad[0, 0].clone()  # save for plotting later
            if not torch.isnan(actions.grad).any():
                # this is definitely needed, judging from some examples I saw where gradient is 50
                torch.nn.utils.clip_grad_norm_(actions, 1.0, norm_type="inf")
                optimizer.step()
            else:
                print("NaN grad!")

            values, indices = cost.min(dim=0)
            best_cost = cost[indices]
            best_actions = actions[indices].unsqueeze(0).clone()

            if metadata is not None:
                metadata["action_history"].append(best_actions)

            if self.visualizer:
                unnormalized_actions = self.normalizer.unnormalize_actions(
                    actions.data
                )
                # self.visualizer.update_values(
                #     best_cost.item(),
                #     unnormalized_actions[0, 0, 0].item(),
                #     unnormalized_actions[0, 0, 1].item(),
                #     a_grad[0].item(),
                #     a_grad[1].item(),
                # )
        self.last_actions = best_actions
        actions = best_actions[:, 0]

        return actions

    def __call__(
        self,
        conditional_state_seq: StateSequence,
        normalize_inputs: bool = False,
        normalize_outputs: bool = False,
        gt_future: Callable[[], StateSequence] = None,
        init_freeze: Optional[torch.Tensor] = None,
        metadata: dict = None,
    ):
        """
        This function is used as a policy in the evaluator.
        - gt_future is ground truth future getter, a function
        that when called returns ground truth future.
        """

        if not self.should_replan():
            actions = self.last_actions[
                :, self.ctr % self.config.planning_freq
            ]

        else:
            if gt_future is not None:
                # gt future is a lambda. this is to save time when we don't
                # plan every step.
                gt_future_seq = gt_future()
                if gt_future_seq is None:
                    self.fm_fallback = True
            else:
                gt_future_seq = None

            if normalize_inputs:
                conditional_state_seq = self.normalizer.normalize_state_seq(
                    conditional_state_seq
                )
                if gt_future_seq is not None:
                    gt_future_seq = self.normalizer.normalize_state_seq(
                        gt_future_seq
                    )
            actions = self.plan(conditional_state_seq, gt_future_seq, metadata)

        self.ctr += 1
        if normalize_outputs:
            actions = self.normalizer.unnormalize_actions(actions.data)
        print("final actions for", self.ctr, "are", actions)

        self.cost.traj_landscape = False
        if self.visualizer is not None:
            self.visualizer.update_plot()

        return actions.detach()


MPCFMPolicy = MPCKMPolicy
# class MPCFMPolicy(torch.nn.Module):
#     @dataclass
#     class Config(configs.ConfigBase):
#         n_iter: int = 10
#         lr: float = 0.1
#         unfold_len: int = 30
#         timestep: float = 0.1
#         update_ref_period: int = 100
#         use_fm: bool = True
#         optimizer: str = "Adam"
#         fm_unfold_variance: float = 1.0
#         fm_unfold_samples: int = 1
#         fm_unfold_samples_agg: str = "max"
#         planning_freq: int = 1
#         lambda_j_mpc: float = 0.0
#         batch_size: int = 1
#         ce_repeat_step: int = 5
#         unfolding_agg: str = "max"
#         save_opt_stats: bool = False

#     OPTIMIZER_DICT = {
#         "SGD": torch.optim.SGD,
#         "Adam": torch.optim.Adam,
#     }

#     def __init__(
#         self,
#         forward_model,
#         cost,
#         normalizer,
#         config,
#         visualizer=None,
#     ):
#         super().__init__()

#         self.cost = cost
#         self.forward_model = forward_model
#         self.normalizer = normalizer
#         self.config = config

#         self.last_actions = None
#         self.visualizer = visualizer
#         self.reset()

#     def reset(self):
#         self.last_actions = None
#         self.ctr = 0
#         self.optimizer_stats = None

#     def unfold_fm(self, images, states, actions, Z=None):
#         """
#         Autoregressively applies fm prediction to get reference images and states.
#             states shape : batch, state_dim
#             actions shape : batch, unfold_len, action_dim
#             Z shape : batch, unfold_len, 32
#         Returns:
#             predicted_images, shape = batch, unfold_len, images channels, images h, images w
#             predicted_states, shape = batch, unfold_len, state_dim
#         """

#         actions_per_fm_timestep = int(0.1 / self.config.timestep)
#         if (
#             actions_per_fm_timestep == 0
#             or self.config.unfold_len % actions_per_fm_timestep != 0
#             or not self.config.use_fm
#         ):
#             ref_states = self.unfold_km(
#                 states[..., -1, :].view(-1, 5), torch.zeros_like(actions)
#             )
#             return (
#                 images[:, -1].repeat(1, self.config.unfold_len, 1, 1, 1),
#                 ref_states,
#             )
#         else:
#             # We make a batch of the same values, but we use different latents.
#             # The motivation is to get multiple fm predictions and plan through that to get
#             # better results that reflect the uncertainty.

#             actions = repeat_batch(actions, self.config.fm_unfold_samples)
#             images = repeat_batch(images, self.config.fm_unfold_samples)
#             states = repeat_batch(states, self.config.fm_unfold_samples)

#             if Z is None:
#                 Z = (
#                     torch.randn(*actions.shape[:2], 32)
#                     * self.config.fm_unfold_variance
#                 )
#                 print("WARN: reinitializing Z")

#             unfolding = self.forward_model.model.unfold(
#                 actions_or_policy=actions,
#                 batch={
#                     "input_images": images.cuda(),
#                     "input_states": states.cuda(),
#                 },
#                 Z=Z,
#             )

#             if self.config.fm_unfold_samples_agg == "max":
#                 ref_images = (
#                     unfolding["pred_images"].max(dim=0, keepdim=True).values
#                 )
#             elif self.config.fm_unfold_samples_agg == "mean":
#                 ref_images = unfolding["pred_images"].mean(dim=0, keepdim=True)
#             elif self.config.fm_unfold_samples_agg == "keep":
#                 ref_images = unfolding["pred_images"]

#             if self.config.fm_unfold_samples_agg == "keep":
#                 ref_states = unfolding["pred_states"]
#             else:
#                 # TODO: this has to also account for the fact that other
#                 # cars are probably moving at the same rate as us.
#                 ref_states = unfolding["pred_states"][:1]

#             return ref_images, ref_states, unfolding["Z"]

#     def __call__(
#         self,
#         conditional_state_seq: StateSequence,
#         normalize_inputs=False,
#         normalize_outputs=False,
#         init=None,
#         metadata=None,
#     ):
#         device = conditional_state_seq.states.device

#         if self.ctr % self.config.planning_freq > 0:
#             actions = self.last_actions[
#                 :, self.ctr % self.config.planning_freq
#             ]

#             if normalize_outputs:
#                 actions = self.normalizer.unnormalize_actions(actions.data)

#             print("final actions for", self.ctr, "are", actions)

#             self.ctr += 1

#             return actions.detach()

#         # if self.ctr == 99:
#         #     dump_dict = dict(images=images, states=states, car_size=car_size,)
#         #     torch.save(dump_dict, "bad_example.dump")

#         if self.visualizer:
#             self.visualizer.step_reset()

#         if normalize_inputs:
#             states = self.normalizer.normalize_states(states.clone())
#             images = self.normalizer.normalize_images(images)
#             car_size = torch.tensor(car_size).unsqueeze(0)

#         full_states = states.unsqueeze(0)
#         full_images = images[:, :3].unsqueeze(0)
#         states = states[..., -1, :].view(-1, 5)
#         images = images[..., -1, :, :, :].view(-1, 1, 4, 117, 24)
#         # if self.last_actions is not None:
#         #     actions = torch.cat(
#         #         (self.last_actions[:, 1:-1], self.last_actions[:, -2].unsqueeze(1).repeat(1, 2, 1)), dim=1
#         #     )
#         #     actions = torch.tensor(actions, requires_grad=True)
#         # else:

#         # Zero normalized maps to slight acceleration when normalized. We make sure we start from
#         # true zero.
#         actions = torch.cat(
#             [
#                 # The first action is always just zeros.
#                 torch.zeros(1, self.config.unfold_len, 2, device=device),
#                 # The rest are random repeated.
#                 torch.randn(
#                     self.config.batch_size - 1, 1, 2, device=device
#                 ).repeat(1, self.config.unfold_len, 1),
#             ],
#             dim=0,
#         )
#         # actions[:, :, 0] = 30
#         # best_actions = torch.zeros_like(actions[:1])
#         actions = torch.randn_like(actions) * 0.01
#         actions[:, :, 1] = 0
#         best_actions = actions[:1]

#         # TODO: remove the changes to init in actions
#         best_cost = 1e10

#         if init is not None:
#             actions[0, 0] = init

#         actions = self.normalizer.normalize_actions(actions)
#         self.cost.traj_landscape = False
#         actions.requires_grad = True

#         def get_cost(
#             actions,
#             Z=None,
#             keep_batch_dim=False,
#             unfolding_agg="max",
#             metadata=None,
#         ):
#             batch_size = actions.shape[0]
#             unfoldings_size = self.config.fm_unfold_samples

#             rep_states = repeat_batch(full_states, batch_size)
#             rep_images = repeat_batch(full_images, batch_size)
#             rep_actions = repeat_batch(actions, unfoldings_size)

#             pred_images, pred_states, Z = self.unfold_fm(
#                 rep_images, rep_states, rep_actions, Z
#             )

#             if metadata is not None:
#                 metadata["pred_images"].append(pred_images.clone().detach())
#                 metadata["pred_states"].append(pred_states.clone().detach())

#             inputs = {
#                 "input_images": repeat_batch(
#                     full_images, batch_size * unfoldings_size
#                 ),
#                 "input_states": repeat_batch(
#                     full_states, batch_size * unfoldings_size
#                 ),
#                 "car_sizes": repeat_batch(
#                     car_size, batch_size * unfoldings_size
#                 ),
#             }
#             predictions = {
#                 "pred_states": pred_states,
#                 "pred_images": pred_images,
#                 "pred_actions": repeat_batch(actions, unfoldings_size),
#                 "Z": Z,
#             }

#             jerk_actions = actions
#             if self.last_actions is not None:
#                 # Cat the last action so we account for what action we performed in the past when calculating jerk.
#                 jerk_actions = torch.cat(
#                     [
#                         repeat_batch(self.last_actions[:, :1], batch_size),
#                         actions,
#                     ],
#                     dim=1,
#                 )

#             gamma_mask = (
#                 torch.tensor(
#                     [
#                         self.cost.config.gamma ** t
#                         for t in range(jerk_actions.shape[1] - 1)
#                     ]
#                 )
#                 .cuda()
#                 .unsqueeze(0)
#             )
#             loss_j = (
#                 (jerk_actions[:, 1:] - jerk_actions[:, :-1])
#                 .norm(2, 2)
#                 .pow(2)
#                 .mul(gamma_mask)
#                 .mean(dim=1)
#             )

#             # costs = self.cost.compute_state_costs_for_training(inputs, pred_images, pred_states, actions, car_size)
#             costs = self.cost.calculate_cost(inputs, predictions)

#             if metadata is not None:
#                 metadata["costs"].append(costs)
#                 metadata["predictions"].append(predictions)

#             result = (
#                 costs["policy_loss"]
#                 + repeat_batch(loss_j, unfoldings_size)
#                 * self.config.lambda_j_mpc
#             )

#             # TODO: double check this is correct.
#             result = result.view(unfoldings_size, batch_size)

#             if unfolding_agg == "mean":
#                 result = result.mean(dim=0)
#             elif unfolding_agg == "max":
#                 result = result.max(dim=0).values

#             if keep_batch_dim:
#                 return result
#             else:
#                 return result.mean()

#         if self.config.optimizer in ["SGD", "Adam"]:
#             optimizer = self.OPTIMIZER_DICT[self.config.optimizer](
#                 (actions,), self.config.lr
#             )

#             if (
#                 self.optimizer_stats is not None
#             ) and self.config.save_opt_stats:
#                 print("loading opt stats")
#                 optimizer.load_state_dict(self.optimizer_stats)

#             # We keep the latent constant to make optimization easier.
#             Z = (
#                 torch.randn(self.config.batch_size, self.config.unfold_len, 32)
#                 * self.config.fm_unfold_variance
#             )
#             Z = repeat_batch(Z, self.config.fm_unfold_samples, interleave=True)

#             for i in range(self.config.n_iter):
#                 optimizer.zero_grad()
#                 if i == self.config.n_iter - 1 and self.visualizer is not None:
#                     self.cost.traj_landscape = True

#                 cost = get_cost(
#                     actions,
#                     Z=Z,
#                     keep_batch_dim=True,
#                     unfolding_agg=self.config.unfolding_agg,
#                     metadata=metadata,
#                 )

#                 if i == self.config.n_iter - 1:
#                     self.cost.traj_landscape = False

#                 cost.mean().backward()
#                 a_grad = actions.grad[0, 0].clone()  # save for plotting later

#                 if not torch.isnan(actions.grad).any():
#                     # this is definitely needed, judging from some examples I saw where gradient is 50
#                     torch.nn.utils.clip_grad_norm_(
#                         actions, 1.0, norm_type="inf"
#                     )
#                     optimizer.step()
#                 else:
#                     print("NaN grad!")

#                 values, indices = cost.min(dim=0)
#                 # if cost[indices] < best_cost:
#                 #     best_actions = actions[indices].unsqueeze(0).clone()
#                 #     best_cost = cost[indices]
#                 best_cost = cost[indices]
#                 best_actions = actions[indices].unsqueeze(0).clone()

#                 if self.visualizer:
#                     unnormalized_actions = self.normalizer.unnormalize_actions(
#                         actions.data
#                     )
#                     self.visualizer.update_values(
#                         best_cost.item(),
#                         unnormalized_actions[0, 0, 0].item(),
#                         unnormalized_actions[0, 0, 1].item(),
#                         a_grad[0].item(),
#                         a_grad[1].item(),
#                     )

#                 self.optimizer_stats = optimizer.state_dict()

#         self.cost.traj_landscape = False

#         if self.visualizer is not None:
#             self.visualizer.update_plot()

#         self.last_actions = best_actions

#         actions = best_actions[:, 0]

#         if normalize_outputs:
#             actions = self.normalizer.unnormalize_actions(actions.data)

#         print("final actions for", self.ctr, "are", actions)

#         self.ctr += 1

#         return actions.detach()

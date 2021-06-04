from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

from ppuu import configs
from ppuu.data.entities import StateSequence
from ppuu.modeling.km import predict_states, predict_states_seq


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


def nan_to_val(x, val=1e8):
    old_shape = x.shape
    x_flat = x.flatten()
    x_flat[torch.isnan(x_flat)] = val
    return x_flat.view(*old_shape)


def flatten_f_unflatten(f, args, dims=2):
    """Flattens first dims in x, applies
    x to the tensor, and unflattens the result to have the
    original first two dims"""
    original_shapes = [
        x.shape if torch.is_tensor(x) else x.images.shape for x in args
    ]
    args_flat = [flatten_dims(x, dims) for x in args]
    f_x_flat = f(*args_flat)
    return unflatten_dims(f_x_flat, original_shapes[0][:dims])


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
        plan_length=30,
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
        self.plan_length = plan_length

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
        actions = a_mu + a_std * torch.randn(
            self.batch_size,
            self.population_size,
            self.plan_length,
            self.a_size,
            device=self.device,
        )
        actions.requires_grad = True

        optimizer = torch.optim.SGD((actions,), self.lr)

        for _ in range(self.n_iter):
            # now we want to get the rewards for those actions.
            costs = get_cost(actions)

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
                self.plan_length,
                self.a_size,
            ).gather(
                dim=1,
                index=topk.view(*topk.shape, 1, 1).repeat(
                    1, 1, self.plan_length, 2
                ),
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
        lambda_action_log_barrier: float = 1.0
        lbfgs_max_iter: int = 10
        lbfgs_limit: float = 10
        fm_unfold_variance: float = 1.0
        fm_unfold_samples: int = 1
        fm_unfold_samples_agg: str = "max"
        planning_freq: int = 1
        lambda_j_mpc: float = 0.0
        batch_size: int = 1
        ce_top_size: int = 10
        ce_gd: bool = True
        ce_variance: float = 4.0
        unfolding_agg: str = "max"
        km_noise: float = 0.0
        save_opt_stats: bool = False
        plan_size: Optional[int] = None
        clip_actions: Optional[float] = None
        greedy_range: Optional[float] = None
        greedy_points: Optional[int] = None
        action_grid_batch: int = 625
        lbfgs_history_size: int = 2

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
        self.optimizer_stats = None

    def unfold_km(self, states, actions):
        """
        Autoregressively applies km state prediction to states with given actions.
            states shape : batch, actions_bsize, state_dim
            actions shape : batch, acitions_bsize, unfold_len, action_dim
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
                    actions[:, :, i],
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
        actions = self._repeat_actions_for_horizon(actions, dim=1)

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

        Z = (
            torch.randn(*actions.shape[:3], 32)
            * self.config.fm_unfold_variance
        ).to(actions.device)

        unfolding = flatten_f_unflatten(
            self.forward_model.unfold,
            (rep_conditional_state_seq.without_ego(), actions, Z),
            dims=2,
        )

        # Depending on aggregation function, we do aggregation
        # along the second dimension, which is the unfoldings dimensions.
        if self.config.fm_unfold_samples_agg == "max":
            images = unfolding.state_seq.images.max(dim=1, keepdim=True).values
        elif self.config.fm_unfold_samples_agg == "mean":
            images = unfolding.state_seq.images.mean(dim=1, keepdim=True)
        elif self.config.fm_unfold_samples_agg == "keep":
            # we still keep the dimension
            images = unfolding.state_seq.images

        # Depending on if we collapse the dim, we need to return different
        # states/car_sizes/ego_car_image.
        if self.config.fm_unfold_samples_agg == "keep":
            result_seq = StateSequence(
                images=images,
                states=unfolding.state_seq.states,
                car_size=unfolding.state_seq.car_size,
                ego_car_image=unfolding.state_seq.ego_car_image,
            )

        else:
            result_seq = StateSequence(
                images=images,
                states=unfolding.state_seq.states[:, :1],
                car_size=unfolding.state_seq.car_size[:, :1],
                ego_car_image=unfolding.state_seq.ego_car_image[:, :1],
            )

        # print(
        #     f"After generating images {result_seq.images.shape=}, {result_seq.states.shape=}"
        # )

        return result_seq

    def should_replan(self) -> bool:
        """Returns true if according to config this step has to be replanned"""
        return self.ctr % self.config.planning_freq == 0

    def _jerk_loss_with_last_action(
        self, actions: torch.Tensor
    ) -> torch.Tensor:
        """Expects tensor of shape
        bsize, action_batch_size, npred, 2
        """
        actions_batch_size = actions.shape[1]
        if self.last_actions is not None:
            # Cat the last action so we account for what action we performed in the past when calculating jerk.
            jerk_actions = torch.cat(
                [
                    flatten_2_dims(
                        repeat_batch(
                            self.last_actions[:, :1].unsqueeze(1),
                            actions_batch_size,
                            dim=1,
                        )
                    ),
                    flatten_2_dims(actions),
                ],
                dim=1,
            )
        else:
            jerk_actions = flatten_2_dims(actions)

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

    def _action_log_barrier(self, actions: torch.Tensor) -> torch.Tensor:
        """actions should be normalized"""
        # This serves as a help to gradient prop if we actually go through the barrier.
        LIMIT = self.config.lbfgs_limit
        # adjusted_actions = (
        #     actions.abs()
        #     - (
        #         actions.abs() - torch.clamp(actions.abs(), max=LIMIT) + 1e-19
        #     ).detach()
        # )
        # log_barrier = -self.config.lambda_action_log_barrier * torch.clamp(
        #     (LIMIT - adjusted_actions).log().mean(dim=-1).mean(dim=-1), max=0
        # )
        # this is for when we actually cross the barrier
        sq_barrier = (
            (torch.clamp(actions.abs() - LIMIT, min=0) ** 2)
            .mean(dim=-1)
            .mean(dim=-1)
        )
        # we clamp to not care if we're far enough.
        return sq_barrier

    def _input_context_cross_product(
        self,
        conditional_state_seq: StateSequence,
        scalar_states: torch.Tensor,
        actions: torch.Tensor,
        context_state_seq: StateSequence,
    ) -> Tuple[StateSequence, torch.Tensor, torch.Tensor, StateSequence]:
        # we repeat all inputs, and repeat interleave all contexts to achieve
        # cross-product.
        # then, we can reshape back to batch size, unfoldings, action_batch_size...
        action_batch_size = actions.shape[1]
        unfoldings_batch_size = context_state_seq.images.shape[1]
        # print(
        #     f"Inside cross product {action_batch_size=}, {unfoldings_batch_size=}"
        # )

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
                interleave=True,
            )
        )
        return (
            cp_conditional_state_seq,
            cp_scalar_states,
            cp_actions,
            cp_context_state_seq,
        )

    def _repeat_actions_for_horizon(self, actions: torch.Tensor, dim=1):
        # If we have fewer actions that unfoldings steps,
        # we repeat the actions to fit the desired length.
        if actions.shape[dim] != self.config.unfold_len:
            assert (
                self.config.unfold_len % actions.shape[dim] == 0
            ), f"can't use specified actions size ({actions.shape[dim]}) with horizin={self.config.unfold_len}"
            repeat_size = self.config.unfold_len // actions.shape[dim]
            actions = actions.repeat_interleave(repeat_size, dim=dim)
        return actions

    def get_cost(
        self,
        conditional_state_seq: StateSequence,
        actions: torch.Tensor,
        context_state_seq: StateSequence,
        unfolding_agg: str = "max",
        metadata: Optional[dict] = None,
    ):
        """Gets the cost of actions.
        actions shape is batch_size, actions_batch_size, npred, 2
        context_state_seq shape is batch_size, unfoldings_size, npred, ...
        the output shape is batch_size, action_batch_size
        """
        batch_size = actions.shape[0]
        action_batch_size = actions.shape[1]
        unfolding_size = context_state_seq.images.shape[1]

        # Repeat actions along dim #2, first two are batch and actions batch.
        actions = self._repeat_actions_for_horizon(actions, dim=2)

        # All conditional states are of shape
        # batch, npred, ...
        rep_conditional_states = repeat_batch(
            conditional_state_seq.states.unsqueeze(1), action_batch_size, dim=1
        )

        # now we have actions and states of shape batch, actions_batch_size, ...
        pred_states = flatten_f_unflatten(
            self.unfold_km,
            (rep_conditional_states[..., -1, :], actions),
            dims=2,
        )

        # Now we need to do a cross product between the tuple
        # (actions, pred_states) and different context images.

        # Now we'd have batch, action_batch_size * unfolding_batch_size, ...
        # shapes, done with repeat/repeat_interleave.
        (
            cross_product_conditional_state_seq,
            cross_product_scalar_states,
            cross_product_actions,
            cross_product_context_state_seq,
        ) = self._input_context_cross_product(
            conditional_state_seq,
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
        if metadata is not None:
            if "costs" not in metadata:
                metadata["costs"] = []
            metadata["costs"].append(costs)

        assert list(costs.total.shape) == [
            batch_size * action_batch_size * unfolding_size
        ]

        total_cost = costs.total.view(
            batch_size, unfolding_size, action_batch_size
        )

        loss_j = (
            self._jerk_loss_with_last_action(actions)
            .view(batch_size, action_batch_size)
            .unsqueeze(1)
        )

        loss_a_log_barrier = (
            self._action_log_barrier(flatten_2_dims(actions))
            .view(batch_size, action_batch_size)
            .unsqueeze(1)
        )

        result = (
            total_cost  # shape bsize, unfoldings_batch, action_batch
            + loss_j * self.config.lambda_j_mpc  # shape bsize, 1, action_batch
            + loss_a_log_barrier
            * self.config.lambda_action_log_barrier  # shape bsize, 1, action_batch
        )

        if unfolding_agg == "mean":
            result = result.mean(dim=1)
        elif unfolding_agg == "max":
            result = result.max(dim=1).values

        assert list(result.shape) == [batch_size, action_batch_size]
        return result

    def _init_optimizer(self, actions: torch.Tensor):
        if self.config.optimizer == "LBFGS":
            optimizer = self.OPTIMIZER_DICT[self.config.optimizer](
                (actions,),
                lr=self.config.lr,
                line_search_fn="strong_wolfe",
                max_iter=self.config.lbfgs_max_iter,
                history_size=self.config.lbfgs_history_size,
            )
        else:
            optimizer = self.OPTIMIZER_DICT[self.config.optimizer](
                (actions,), self.config.lr
            )
        if (self.optimizer_stats is not None) and self.config.save_opt_stats:
            optimizer.load_state_dict(self.optimizer_stats)
        return optimizer

    def _build_closure(
        self,
        conditional_state_seq: StateSequence,
        future_context_state_seq: StateSequence,
        actions: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        gd: bool = True,
        metadata: Optional[dict] = None,
        keep_dim: bool = False,
    ):
        def closure(actions: torch.Tensor):
            if optimizer is not None:
                optimizer.zero_grad()
            cost = self.get_cost(
                conditional_state_seq,
                actions,
                context_state_seq=future_context_state_seq,
                unfolding_agg=self.config.unfolding_agg,
                metadata=metadata,
            )
            self._last_cost = cost.data

            if metadata is not None:
                if "action_history" not in metadata:
                    metadata["action_history"] = [actions.clone().detach()]
                else:
                    metadata["action_history"].append(actions.clone().detach())

                if "cost_history" not in metadata:
                    metadata["cost_history"] = [cost.clone().detach()]
                else:
                    metadata["cost_history"].append(cost.clone().detach())

            if gd:
                cost.sum().backward()

                if not torch.isnan(actions.grad).any():
                    # this is definitely needed, judging from some examples I saw where gradient is 50
                    if not self.config.optimizer == "LBFGS":
                        torch.nn.utils.clip_grad_norm_(
                            actions, 1.0, norm_type="inf"
                        )
                else:
                    print(f"NaN grad! {actions.grad.max()=}, {actions.max()=}")
            if keep_dim:
                return cost
            else:
                return cost.sum()

        if actions is None:
            return closure
        else:
            return lambda: closure(actions)

    def _optimize_one_step(
        self,
        optimizer: torch.optim.Optimizer,
        conditional_state_seq: StateSequence,
        future_context_state_seq: StateSequence,
        actions: torch.Tensor,
        metadata: Optional[dict] = None,
    ):
        closure = self._build_closure(
            conditional_state_seq,
            future_context_state_seq,
            actions,
            optimizer,
            metadata=metadata,
        )

        optimizer.step(closure)

        a_grad = actions.grad[0, 0, 0].clone()  # save for plotting later
        if self.config.clip_actions is not None:
            # in case it explodes
            actions.data = torch.clamp(
                actions.data,
                min=-self.config.clip_actions,
                max=self.config.clip_actions,
            )

        if self.visualizer:
            unnormalized_actions = self.normalizer.unnormalize_actions(
                actions.data
            )
            self.visualizer.update_values(
                self._last_cost[0, 0].item(),
                unnormalized_actions[0, 0, 0, 0].item(),
                unnormalized_actions[0, 0, 0, 1].item(),
                a_grad[0].item(),
                a_grad[1].item(),
            )

    @torch.no_grad()
    def _greedy_optimize(
        self,
        conditional_state_seq: StateSequence,
        best_actions: torch.Tensor,
        gt_future_state_seq: Optional[StateSequence],
    ):
        future_context_state_seq = self._get_context(
            conditional_state_seq,
            best_actions,
            0,
            None,
            gt_future_state_seq,
        )
        xx, yy, costs = self._try_action_grid(
            conditional_state_seq,
            future_context_state_seq,
            points=self.config.greedy_points,
            search_range=self.config.greedy_range,
        )
        batch_size = xx.shape[0]
        xx = xx.view(batch_size, -1)
        yy = yy.view(batch_size, -1)
        costs = costs.view(batch_size, -1)
        costs = nan_to_val(costs)
        min_idx = costs.min(dim=-1).indices.unsqueeze(-1)
        acc = torch.gather(xx, 1, min_idx)
        turns = torch.gather(yy, 1, min_idx)
        actions = torch.stack([acc, turns], dim=-1)
        self.last_actions = actions.repeat(1, self.config.unfold_len, 1)
        return actions[:, 0]

    def _ce_optimize(
        self,
        conditional_state_seq: StateSequence,
        best_actions: torch.Tensor,
        gt_future_state_seq: Optional[StateSequence],
        metadata: Optional[dict] = None,
    ):
        future_context_state_seq = self._get_context(
            conditional_state_seq,
            best_actions,
            0,
            None,
            gt_future_state_seq,
        )
        closure = self._build_closure(
            conditional_state_seq,
            future_context_state_seq,
            gd=False,
            keep_dim=True,
            metadata=metadata,
        )
        actions = CE(
            horizon=self.config.unfold_len,
            n_iter=self.config.n_iter,
            population_size=self.config.batch_size,
            top_size=self.config.ce_top_size,
            variance=self.config.ce_variance,
            gd=self.config.ce_gd,
            lr=self.config.lr,
            plan_length=self.config.plan_size,
        ).plan(closure)
        self.last_actions = actions.repeat(
            1, self.config.unfold_len // self.config.plan_size, 1
        ).detach()
        return actions[:, 0]

    def _init_actions(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Initializes actions.
        If config says we should only have one action for the entire prediction,
        we'll create just one action, otherwise we create unfolding_len actions.
        The output is normalized actions.
        """
        if self.config.plan_size is None:
            actions_size = self.config.unfold_len
        else:
            actions_size = self.config.plan_size

        actions = torch.cat(
            [
                # The first action is always just zeros.
                self.normalizer.normalize_actions(
                    torch.zeros(batch_size, 1, actions_size, 2, device=device)
                ),
                # The rest are random repeated.
                torch.randn(
                    batch_size, self.config.batch_size - 1, 1, 2, device=device
                ).repeat(1, 1, actions_size, 1),
            ],
            dim=1,
        )
        # actions[:, :, :, 1] = 0
        return actions

    def _select_best_actions(self, actions: torch.Tensor, costs: torch.Tensor):
        values, indices = nan_to_val(costs, 1e8).min(dim=1)
        flat_actions = actions.view(
            *actions.shape[:2], -1
        )  # bsize, 1, npred * 2
        # gather indices should be of shape batch size, 1, npred * 2
        gather_idx = indices.view(actions.shape[0], 1, 1).expand(
            actions.shape[0], 1, flat_actions.shape[2]
        )
        gathered_flat_actions = torch.gather(flat_actions, 1, gather_idx)
        # the result we get should have shape batch_size, npred, 2
        return (
            gathered_flat_actions.view(actions.shape[0], *actions.shape[2:]),
            values,
        )

    def _get_context(
        self,
        conditional_state_seq: StateSequence,
        best_actions: torch.Tensor,
        i: int,
        current_context_state_seq: Optional[StateSequence],
        gt_future_state_seq: Optional[StateSequence],
    ):
        # print("getting context")
        if gt_future_state_seq is not None:
            # print("gt future")
            return gt_future_state_seq
        else:
            if (
                current_context_state_seq is not None
                and i % self.config.update_ref_period != 0
            ):
                # print("current")
                return current_context_state_seq
            else:
                # print("gen")
                return self.unfold_fm(conditional_state_seq, best_actions)

    def plan(
        self,
        conditional_state_seq: StateSequence,
        gt_future_state_seq: Optional[StateSequence] = None,
        metadata: Optional[dict] = None,
        full_plan: bool = False,
    ) -> torch.Tensor:
        """Plans given conditional state sequence, and optionally gt_future_state_seq.
        Everything is normalized in input and output.
        """

        if self.visualizer:
            self.visualizer.step_reset()

        actions = self._init_actions(
            batch_size=conditional_state_seq.states.shape[0],
            device=conditional_state_seq.states.device,
        )
        actions.requires_grad = True

        best_actions = actions[:, 0]  # we take the first action initially
        best_cost = 1e10

        if metadata is not None:
            metadata["action_history"] = []
            metadata["cost_history"] = []

        if self.config.optimizer == "greedy":
            return self._greedy_optimize(
                conditional_state_seq,
                best_actions.detach(),
                gt_future_state_seq,
            )
        elif self.config.optimizer == "CE":
            return self._ce_optimize(
                conditional_state_seq,
                best_actions.detach(),
                gt_future_state_seq,
                metadata=metadata,
            )

        optimizer = self._init_optimizer(actions)

        future_context_state_seq = None

        for i in range(self.config.n_iter):
            future_context_state_seq = self._get_context(
                conditional_state_seq,
                best_actions,
                i,
                future_context_state_seq,
                gt_future_state_seq,
            )

            self._optimize_one_step(
                optimizer,
                conditional_state_seq,
                future_context_state_seq,
                actions,
                metadata=metadata,
            )

        # Calculate the cost again and choose the best one.
        with torch.no_grad():
            cost = self.get_cost(
                conditional_state_seq,
                actions,
                future_context_state_seq,
                unfolding_agg=self.config.unfolding_agg,
            )
        best_actions, best_cost = self._select_best_actions(
            actions.data, cost.data
        )

        # check that we got rid of actions_batch_size
        assert list(best_actions.shape) == [
            actions.shape[0],
            *actions.shape[2:],
        ]

        self.last_actions = self._repeat_actions_for_horizon(
            best_actions, dim=1
        )
        if full_plan:
            actions = best_actions
        else:
            actions = best_actions[:, 0]
        self.optimizer_stats = optimizer.state_dict()

        return actions

    def __call__(
        self,
        conditional_state_seq: StateSequence,
        normalize_inputs: bool = False,
        normalize_outputs: bool = False,
        gt_future: Optional[
            Union[Callable[[], StateSequence], StateSequence]
        ] = None,
        metadata: dict = None,
        full_plan: bool = False,
    ):
        """
        This function is used as a policy in the evaluator.
        - gt_future is ground truth future getter, a function
        that when called returns ground truth future.
        """

        if (
            not full_plan
            and self.last_actions is not None
            and not self.should_replan()
        ):
            actions = self.last_actions[
                :, self.ctr % self.config.planning_freq
            ]

        else:
            if gt_future is not None:
                # gt future is a lambda. this is to save time when we don't
                # plan every step.
                if callable(gt_future):
                    gt_future_seq = gt_future()
                else:
                    gt_future_seq = gt_future
                if gt_future_seq is not None:
                    gt_future_seq = gt_future_seq.map(lambda x: x.unsqueeze(1))
                else:
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
            actions = self.plan(
                conditional_state_seq,
                gt_future_seq,
                metadata=metadata,
                full_plan=full_plan,
            )

        self.ctr += 1
        if normalize_outputs:
            actions = self.normalizer.unnormalize_actions(actions.data)
        # print("final actions for", self.ctr, "are", actions)
        return actions.detach()

    @torch.no_grad()
    def _try_action_grid(
        self,
        conditional_state_seq: StateSequence,
        future_context_state_seq: StateSequence,
        points: int = 25,
        search_range: float = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constructs a matrix of all possible actions and their corresponding
        costs.
        Returns a tuple: accelerations, turns, and costs
        """
        x = torch.linspace(-1, 1, points) * (-search_range)
        # later we draw it and y axis will be flipped
        y = torch.linspace(-1, 1, points) * search_range
        xx, yy = torch.meshgrid([x, y])
        actions = (
            torch.stack([xx, yy], dim=-1)
            .view(-1, 2)
            .to(conditional_state_seq.images.device)
        )
        batch_size = conditional_state_seq.images.shape[0]
        actions = (
            actions.unsqueeze(0)
            .unsqueeze(-2)
            .repeat(batch_size, 1, self.config.unfold_len, 1)
        )
        # here we split the computations into batches of 625.
        # and process them sequentially.
        n_batches = (
            actions.shape[1] + self.config.action_grid_batch - 1
        ) // self.config.action_grid_batch
        costs_batches = []
        for i in range(n_batches):
            batch_index_begin = i * self.config.action_grid_batch
            batch_index_end = (i + 1) * self.config.action_grid_batch
            batch_actions = actions[:, batch_index_begin:batch_index_end]
            batch_cost = self.get_cost(
                conditional_state_seq,
                batch_actions,
                future_context_state_seq,
                unfolding_agg=self.config.unfolding_agg,
            )
            costs_batches.append(batch_cost)

        cost = torch.cat(costs_batches, dim=1)

        xx = xx.unsqueeze(0).repeat(batch_size, 1, 1).to(actions.device)
        yy = yy.unsqueeze(0).repeat(batch_size, 1, 1).to(actions.device)
        cost = cost.view(xx.shape)
        return xx, yy, cost


MPCFMPolicy = MPCKMPolicy

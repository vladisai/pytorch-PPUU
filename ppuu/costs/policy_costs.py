"""Costs calculation for policy. Calculates uncertainty and state costs.
"""
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from ppuu import configs
from ppuu.data.entities import StateSequence


def repeat_expand(x: torch.Tensor, k: int) -> torch.Tensor:
    """Similar to repeat function along the first dimenstion, but
    implemented using expand. Used in the uncertainty estimation.
    """
    return (
        x.unsqueeze(0).expand(k, *x.shape).contiguous().view(-1, *x.shape[1:])
    )


class PolicyCostBase:
    """Base class for policy costs"""

    def calculate_cost(
        self,
        conditional_state_seq: StateSequence,
        actions: torch.Tensor,
        predicted_state_seq: StateSequence,
    ):
        """Base function used to calculate the cost"""
        raise NotImplementedError()


class PolicyCost(PolicyCostBase):
    """Vanilla cost function. Uses max to get the closest point to the ego car."""

    @dataclass
    class Config(configs.ConfigBase):
        """Configuration of cost calculation"""

        u_reg: float = 0.05
        lambda_a: float = 0.0
        lambda_j: float = 0.0
        lambda_l: float = 0.2
        lambda_o: float = 1.0
        lambda_p: float = 1.0
        gamma: float = 0.99
        u_hinge: float = 0.5
        dreaming_z_reg: float = 0.1
        uncertainty_n_pred: int = 30
        uncertainty_n_models: int = 10
        uncertainty_n_batches: int = 100
        artifact_power: int = 2
        safe_factor: float = 1.5
        skip_contours: bool = True

    class StateCosts(NamedTuple):
        """Tuple to store the result of state cost calculation.
        Total costs are tensors of shape bsize.
        The rest are of shape bsize, npred.
        Costs associated with masks are state costs.
        """

        total_proximity: torch.Tensor
        total_lane: torch.Tensor
        total_offroad: torch.Tensor
        proximity: torch.Tensor
        lane: torch.Tensor
        offroad: torch.Tensor

    class Cost(NamedTuple):
        """Tuple to store the full result of the cost calculation."""

        state: Any  # PolicyCost.StateCosts  # noqa
        uncertainty: torch.Tensor
        action: torch.Tensor
        jerk: torch.Tensor
        total: torch.Tensor

    def __init__(self, config, forward_model, normalizer):
        self.config = config
        self.forward_model = forward_model
        self.normalizer = normalizer
        self.traj_landscape = False
        self.gamma = None

    def build_car_proximity_mask(
        self,
        state_seq: StateSequence,
        unnormalize: bool = False,
    ) -> torch.Tensor:
        """Builds mask used for car proximity cost.
        Returns a tensor (bsize, npred, crop_h, crop_w).
        """
        SCALE = 0.25
        safe_factor = 1.5
        bsize, npred, nchannels, crop_h, crop_w = state_seq.images.size()
        images = state_seq.images.view(
            bsize * npred, nchannels, crop_h, crop_w
        )
        states = state_seq.states.view(bsize * npred, 5).clone()

        if unnormalize:
            states = self.normalizer.unnormalize_states(states)

        speed = states[:, 4] * SCALE  # pixel/s
        width, length = (
            state_seq.car_size[:, 0],
            state_seq.car_size[:, 1],
        )  # feet
        width = width * SCALE * (0.3048 * 24 / 3.7)  # pixels
        length = length * SCALE * (0.3048 * 24 / 3.7)  # pixels

        safe_distance = (
            torch.abs(speed) * safe_factor + (1 * 24 / 3.7) * SCALE
        )  # plus one metre (TODO change)

        # Compute x/y minimum distance to other vehicles (pixel version)
        # Account for 1 metre overlap (low data accuracy)
        alpha = 1 * SCALE * (24 / 3.7)  # 1 m overlap collision
        # Create separable proximity mask

        max_x = torch.ceil((crop_h - torch.clamp(length - alpha, min=0)) / 2)
        max_y = torch.ceil((crop_w - torch.clamp(width - alpha, min=0)) / 2)
        max_x = (
            max_x.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(images.device)
        )
        max_y = (
            max_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(images.device)
        )

        min_x = torch.clamp(max_x - safe_distance, min=0)
        min_y = torch.ceil(
            crop_w / 2 - width
        )  # assumes other._width / 2 = self._width / 2
        min_y = (
            min_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(images.device)
        )

        x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2
        x_filter = (
            x_filter.unsqueeze(0)
            .expand(bsize * npred, crop_h)
            .type(state_seq.car_size.type())
            .to(images.device)
        )
        x_filter = torch.min(
            x_filter, max_x.view(bsize * npred, 1).expand(x_filter.size())
        )
        x_filter = torch.max(x_filter, min_x.view(bsize * npred, 1))

        x_filter = (x_filter - min_x.view(bsize * npred, 1)) / (
            max_x - min_x
        ).view(bsize * npred, 1)
        y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
        y_filter = (
            y_filter.view(1, crop_w)
            .expand(bsize * npred, crop_w)
            .type(state_seq.car_size.type())
            .to(images.device)
        )
        y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
        y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
        y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (
            max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1)
        )
        x_filter = x_filter.to(images.device)
        y_filter = y_filter.to(images.device)
        proximity_mask = torch.bmm(
            x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w)
        )
        proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
        return proximity_mask

    def build_lane_offroad_mask(
        self, state_seq: StateSequence
    ) -> torch.Tensor:
        """Builds mask used for calculating lane and offroad costs."""
        SCALE = 0.25
        bsize, npred, nchannels, crop_h, crop_w = state_seq.images.size()
        device = state_seq.images.device

        width, length = (
            state_seq.car_size[:, 0],
            state_seq.car_size[:, 1],
        )  # feet
        width = width * SCALE * (0.3048 * 24 / 3.7)  # pixels
        length = length * SCALE * (0.3048 * 24 / 3.7)  # pixels

        # Create separable proximity mask
        width.fill_(24 * SCALE / 2)

        max_x = torch.ceil((crop_h - length) / 2)
        #    max_y = torch.ceil((crop_w - width) / 2)
        max_y = torch.ceil(torch.zeros(width.size()).fill_(crop_w) / 2)
        max_x = (
            max_x.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(device)
        )
        max_y = (
            max_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(device)
        )
        min_y = torch.ceil(
            crop_w / 2 - width
        )  # assumes other._width / 2 = self._width / 2
        min_y = (
            min_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(device)
        )
        x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2

        x_filter = (
            x_filter.unsqueeze(0)
            .expand(bsize * npred, crop_h)
            .type(state_seq.car_size.type())
            .to(device)
        )

        x_filter = torch.min(
            x_filter, max_x.view(bsize * npred, 1).expand(x_filter.size())
        )
        x_filter = (
            x_filter == max_x.unsqueeze(1).expand(x_filter.size())
        ).float()

        y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
        y_filter = (
            y_filter.view(1, crop_w)
            .expand(bsize * npred, crop_w)
            .type(state_seq.car_size.type())
            .to(device)
        )
        #    y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
        y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
        y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (
            max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1)
        )
        x_filter = x_filter.to(device)
        y_filter = y_filter.to(device)
        x_filter = x_filter.type(y_filter.type())
        proximity_mask = torch.bmm(
            x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w)
        )
        proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
        return proximity_mask

    def _multiply_masks(
        self, images: torch.Tensor, proximity_mask: torch.Tensor
    ) -> torch.Tensor:
        """Given an image of shape bsize, npred, height, width and a mask
        of same shape, multiplies them, and finds max across images.
        Returns tensor of shape bsize, npred.
        """
        return torch.max(
            (proximity_mask * images.float()).view(*images.shape[:2], -1), 2
        )[0].view(*images.shape[:2])

    def compute_lane_cost(
        self, state_seq: StateSequence, proximity_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculates lane cost for each predicted image.
        Returns a tensor of shape (batch size, npred).
        """
        return self._multiply_masks(state_seq.images[:, :, 0], proximity_mask)

    def compute_offroad_cost(
        self, state_seq: StateSequence, proximity_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the offroad cost using the provided mask
        Returns a tensor of shape (batch_size, npred).
        """
        return self._multiply_masks(state_seq.images[:, :, 2], proximity_mask)

    def compute_proximity_cost(
        self,
        state_seq: StateSequence,
        proximity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the proximity cost using the provided mask
        Returns a tensor of shape (batch_size, npred).
        """
        return self._multiply_masks(state_seq.images[:, :, 1], proximity_mask)

    def compute_uncertainty_batch(
        self,
        conditional_state_seq: StateSequence,
        actions: torch.Tensor,
        Z: torch.Tensor = None,
        estimation: bool = True,
    ):
        """Estimates prediction uncertainty using dropout."""
        if estimation:
            torch.set_grad_enabled(False)

        actions = actions[..., : self.config.uncertainty_n_pred, :]
        (
            bsize,
            ncond,
            channels,
            height,
            width,
        ) = conditional_state_seq.images.shape
        device = actions.device

        if Z is None:
            Z = self.forward_model.sample_z(
                bsize, self.config.uncertainty_n_pred
            ).to(device)
            Z = Z.view(bsize, self.config.uncertainty_n_pred, -1)

        # We repeat everything to run multiple prediction in the forward model.
        # Dropout will make the predictions different, giving us a way
        # to estimate uncertainty.
        Z_rep = Z.unsqueeze(0)
        Z_rep = repeat_expand(Z, self.config.uncertainty_n_models)
        rep_conditional_state_seq = conditional_state_seq.map(
            lambda x: repeat_expand(
                x, self.config.uncertainty_n_models
            ).clone()
        )
        actions = repeat_expand(actions, self.config.uncertainty_n_models)

        original_value = self.forward_model.training  # to switch back later
        # turn on dropout, for uncertainty estimation
        self.forward_model.train()
        predictions = self.forward_model.unfold(
            rep_conditional_state_seq,
            actions.clone(),
            Z=Z_rep.clone(),
        )
        self.forward_model.train(original_value)

        costs = self.compute_state_costs_for_uncertainty(
            predictions.state_seq,
        )

        pred_costs = (
            self.config.lambda_p * costs.proximity
            + self.config.lambda_l * costs.lane
            + self.config.lambda_o * costs.offroad
        )
        pred_costs = pred_costs.view(
            self.config.uncertainty_n_models,
            bsize,
            self.config.uncertainty_n_pred,
            -1,
        )

        predicted_state_seq = predictions.state_seq.map(
            lambda x: x.view(
                self.config.uncertainty_n_models, bsize, *x.shape[1:]
            ).clone()
        )
        # use variance rather than standard deviation, since it is not
        # differentiable at 0 due to sqrt
        flat_shape = (
            self.config.uncertainty_n_models,
            bsize,
            self.config.uncertainty_n_pred,
            -1,
        )
        pred_images_var = torch.var(
            predicted_state_seq.images.view(*flat_shape), 0
        ).mean(2)
        pred_states_var = torch.var(
            predicted_state_seq.states.view(*flat_shape), 0
        ).mean(2)
        pred_costs_var = torch.var(pred_costs, 0).mean(2)
        pred_costs_mean = torch.mean(pred_costs, 0)

        if not estimation:
            # This is the uncertainty loss of different terms together.
            u_loss_costs = torch.relu(
                (pred_costs_var - self.u_costs_mean) / self.u_costs_std
                - self.config.u_hinge
            )
            # u_loss_states = torch.relu(
            #     (pred_states_var - self.u_states_mean) / self.u_states_std
            #     - self.config.u_hinge
            # )
            u_loss_images = torch.relu(
                (pred_images_var - self.u_images_mean) / self.u_images_std
                - self.config.u_hinge
            )
            total_u_loss = (
                u_loss_costs.mean()
                # + u_loss_states.mean()
                + u_loss_images.mean()
            )
        else:
            total_u_loss = None
            # We disabled gradients earlier for the estimation case.
            torch.set_grad_enabled(True)

        return dict(
            pred_images_var=pred_images_var,
            pred_states_var=pred_states_var,
            pred_costs_var=pred_costs_var,
            pred_costs_mean=pred_costs_mean,
            total_u_loss=total_u_loss,
        )

    def calculate_uncertainty_cost(
        self,
        conditional_state_seq: StateSequence,
        actions: torch.Tensor,
        Z: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.config.u_reg > 0:
            result = self.compute_uncertainty_batch(
                conditional_state_seq,
                actions,
                Z=Z,
                estimation=False,
            )["total_u_loss"]
        else:
            result = torch.tensor(0.0)
        return result

    def estimate_uncertainty_stats(
        self, dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Computes uncertainty estimates for the ground truth actions in the
        training set. This will give us an idea of what normal ranges are using
        actions the forward model was trained on.
        """
        u_images, u_states, u_costs = [], [], []
        data_iter = iter(dataloader)
        device = next(self.forward_model.parameters()).device
        for i in range(self.config.uncertainty_n_batches):
            print(
                (
                    f"[estimating normal uncertainty ranges:"
                    f" {i / self.config.uncertainty_n_batches:2.1%}]"
                ),
                end="\r",
            )
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = batch.to(device)
            result = self.compute_uncertainty_batch(
                batch.conditional_state_seq,
                batch.target_action_seq,
                estimation=True,
            )
            u_images.append(result["pred_images_var"])
            u_states.append(result["pred_states_var"])
            u_costs.append(result["pred_costs_var"])

        print("[estimating normal uncertainty ranges: 100.0%]")

        u_images = torch.stack(u_images).view(
            -1, self.config.uncertainty_n_pred
        )
        u_states = torch.stack(u_states).view(
            -1, self.config.uncertainty_n_pred
        )
        u_costs = torch.stack(u_costs).view(-1, self.config.uncertainty_n_pred)

        self.u_images_mean = u_images.mean(0)
        self.u_states_mean = u_states.mean(0)
        self.u_costs_mean = u_costs.mean(0)

        self.u_images_std = u_images.std(0)
        self.u_states_std = u_states.std(0)
        self.u_costs_std = u_costs.std(0)

    def compute_state_costs(
        self, state_seq: StateSequence
    ) -> Any:  # PolicyCost.StateCosts:  # noqa
        """Costs associated with masks are state costs"""
        artifact_reduced_state_seq = StateSequence(
            state_seq.images ** self.config.artifact_power, *state_seq[1:]
        )
        car_proximity_mask = self.build_car_proximity_mask(
            artifact_reduced_state_seq, unnormalize=True
        )
        lane_offroad_mask = self.build_lane_offroad_mask(
            artifact_reduced_state_seq
        )
        proximity_cost = self.compute_proximity_cost(
            artifact_reduced_state_seq,
            car_proximity_mask,
        )
        lane_cost = self.compute_lane_cost(
            artifact_reduced_state_seq,
            lane_offroad_mask,
        )
        offroad_cost = self.compute_offroad_cost(
            artifact_reduced_state_seq,
            lane_offroad_mask,
        )

        lane_total = torch.mean(self.apply_gamma(lane_cost))
        offroad_total = torch.mean(self.apply_gamma(offroad_cost))
        proximity_total = torch.mean(self.apply_gamma(proximity_cost))
        return PolicyCost.StateCosts(
            total_proximity=proximity_total,
            total_lane=lane_total,
            total_offroad=offroad_total,
            proximity=proximity_cost,
            offroad=offroad_cost,
            lane=lane_cost,
        )

    def compute_state_costs_for_uncertainty(
        self, state_seq: StateSequence
    ) -> Any:  # PolicyCost.StateCost:  # noqa
        return self.compute_state_costs(state_seq)

    def compute_state_costs_for_training(
        self,
        conditional_state_seq: StateSequence,
        actions: torch.Tensor,
        predicted_state_seq: StateSequence,
    ):
        return self.compute_state_costs(predicted_state_seq)

    def compute_state_costs_for_z(self, state_seq: StateSequence):
        return self.compute_state_costs(state_seq)

    def compute_combined_loss(
        self,
        state: Any,  # PolicyCost.StateCosts,
        uncertainty: torch.Tensor,
        action: torch.Tensor,
        jerk: torch.Tensor,
    ) -> torch.Tensor:
        """Uses the config coefficients to calculated combined loss from
        the components.
        """
        return (
            self.config.lambda_p * state.total_proximity
            + self.config.lambda_l * state.total_lane
            + self.config.lambda_o * state.total_offroad
            + self.config.u_reg * uncertainty
            + self.config.lambda_a * action
            + self.config.lambda_j * jerk
        )

    def calculate_jerk_loss(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[1] > 1:
            loss_j = (
                (actions[:, 1:] - actions[:, :-1])
                .norm(2, 2)
                .pow(2)
                .view(actions.shape[0], -1)
                .mean(dim=-1)
            )
        else:
            loss_j = torch.zeros(actions.shape[0], device=actions.device)
        return loss_j

    def calculate_action_loss(self, actions: torch.Tensor) -> torch.Tensor:
        return (
            (actions).norm(2, 2).pow(2).view(actions.shape[0], -1).mean(dim=-1)
        )

    def calculate_cost(
        self,
        conditional_state_seq: StateSequence,
        actions: torch.Tensor,
        predicted_state_seq: StateSequence,
    ):  # -> PolicyCost.Cost
        u_loss = self.calculate_uncertainty_cost(
            conditional_state_seq, actions
        )
        loss_a = self.calculate_action_loss(actions)
        loss_j = self.calculate_jerk_loss(actions)
        state_cost = self.compute_state_costs_for_training(
            conditional_state_seq,
            actions,
            predicted_state_seq,
        )
        total = self.compute_combined_loss(
            state=state_cost, uncertainty=u_loss, action=loss_a, jerk=loss_j
        )
        return PolicyCost.Cost(
            state=state_cost,
            uncertainty=u_loss,
            action=loss_a,
            jerk=loss_j,
            total=total,
        )

    def apply_gamma(self, loss_over_time: torch.Tensor) -> torch.Tensor:
        if (
            self.gamma is None
            or self.gamma.shape[-1] != loss_over_time.shape[-1]
        ):
            self.gamma = (
                torch.tensor(
                    [
                        self.config.gamma ** t
                        for t in range(loss_over_time.shape[-1])
                    ]
                )
                .to(loss_over_time.device)
                .unsqueeze(0)
            )
        return loss_over_time * self.gamma

    # def calculate_z_cost(self, inputs, predictions, original_z=None):
    #     u_loss = self.calculate_uncertainty_cost(inputs, predictions)
    #     proximity_loss = self.compute_state_costs_for_z(
    #         predictions["pred_images"],
    #         predictions["pred_states"],
    #         inputs["car_sizes"],
    #     )["proximity_loss"]
    #     result = self.compute_combined_loss(
    #         proximity_loss=-1 * proximity_loss,
    #         uncertainty_loss=u_loss,
    #         lane_loss=0,
    #         action_loss=0,
    #         jerk_loss=0,
    #         offroad_loss=0,
    #     )
    #     z_reg = torch.tensor(0)
    #     if original_z is not None:
    #         z_reg = self.config.dreaming_z_reg * (
    #             (predictions["Z"] - original_z).norm(2, -1).mean()
    #             / predictions["Z"].shape[-1]
    #         )
    #         result += z_reg
    #     components = dict(
    #         proximity_loss=proximity_loss, u_loss=u_loss, z_reg=z_reg
    #     )
    #     return result, components

    # def get_grad_vid(self, policy_model, batch, device="cuda"):
    #     input_images = batch["input_images"].clone()
    #     input_states = batch["input_states"].clone()
    #     car_sizes = batch["car_sizes"].clone()

    #     input_images = input_images.clone().float().div_(255.0)
    #     input_states = self.normalizer.normalize_states(input_states)
    #     if input_images.dim() == 4:  # if processing single vehicle
    #         input_images = input_images.to(device).unsqueeze(0)
    #         input_states = input_states.to(device).unsqueeze(0)
    #         car_sizes = car_sizes.to(device).unsqueeze(0)

    #     input_images.requires_grad = True
    #     input_states.requires_grad = True
    #     input_images.retain_grad()
    #     input_states.retain_grad()

    #     costs = self.compute_state_costs(input_images, input_states, car_sizes)
    #     combined_loss = self.compute_combined_loss(
    #         proximity_loss=costs["proximity_loss"],
    #         uncertainty_loss=0,
    #         lane_loss=costs["lane_loss"],
    #         action_loss=0,
    #         jerk_loss=0,
    #         offroad_loss=costs["offroad_loss"],
    #     )
    #     combined_loss.backward()
    #     return input_images.grad[:, :, :3].abs().clamp(max=1.0)

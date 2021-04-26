"""Cost model that uses the kinematic model.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch

from ppuu.costs.policy_costs_continuous import PolicyCost, PolicyCostContinuous
from ppuu.data.dataloader import UnitConverter
from ppuu.data.entities import StateSequence


class AggregationFunction:
    """An aggregation function that given an elementwise product
    of mask and the image will give the combined value.
    """

    def __init__(self, s):
        self.s = s

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Forwards arguments to the respective function."""
        if self.s == "sum":
            return torch.sum(*args, **kwargs)
        elif self.s.startswith("logsumexp"):
            beta = float(self.s.split("-")[1])
            return 1 / beta * torch.logsumexp(args[0] * beta, *args[1:])
        else:
            return torch.logsumexp(*args, **kwargs)


def coordinate_curl(
    xx: torch.Tensor, yy: torch.Tensor, radii: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given two matrices of coordinates, x and y, curl them around the given radii
    Returns a tuple of coordinates, xx and yy.
    """
    center_y = radii.view(*xx.shape[:-2], 1, 1)
    center_x = torch.zeros_like(center_y)

    r = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    x1 = xx - center_x

    is_to_the_right = (center_y > 0).float() * 2 - 1
    y1 = (center_y - yy) * is_to_the_right

    alpha = torch.atan2(x1, y1)
    ll = alpha * r
    yy = center_y - is_to_the_right * r
    xx = center_x + ll
    return xx, yy


def coordinate_rotate(
    xx: torch.Tensor, yy: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given two matrices of coordinates, x and y, and a vector that corresponds
    to cosine and sine of the rotation angle, rotate all points in xx and yy.
    Returns a tuple of coordinates, rotated xx and yy.
    """
    c, s = dx, dy
    c = c.view(*xx.shape[:-2], 1, 1)
    s = s.view(*xx.shape[:-2], 1, 1)
    x_prime = c * xx - s * yy
    y_prime = s * xx + c * yy
    return x_prime, y_prime


def coordinate_rotate_matrix(xx, yy, matrix):
    """xx, yy - [*dims, H, W]
    matrix - [*dims, 2, 2]
    """
    xy = torch.stack([xx, yy], dim=-1)
    xy = xy.view(-1, 2, 1)
    matrix = matrix.unsqueeze(-3).unsqueeze(-3)
    repeat = [1] * len(matrix.shape)
    repeat[-3] = xx.shape[-1]
    repeat[-4] = xx.shape[-2]
    matrix = matrix.repeat(*repeat).view(-1, 2, 2)
    rotated = torch.bmm(matrix, xy)
    xx = rotated[:, 0].view(*xx.shape)
    yy = rotated[:, 1].view(*yy.shape)
    return xx, yy


def rotation_matrix(v1, v2):
    s1 = v1.shape
    v1 = v1.view(-1, 2)
    v2 = v2.view(-1, 2)
    rot = torch.stack(
        [
            v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1],
            v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1],
            -v2[:, 0] * v1[:, 1] + v1[:, 0] * v2[:, 1],
            v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1],
        ],
    )
    rot = rot.view(2, 2, -1).permute(2, 0, 1).reshape(*s1[:-1], 2, 2)
    return rot


def coordinate_shift(
    xx: torch.Tensor,
    yy: torch.Tensor,
    shift_x: torch.Tensor,
    shift_y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shifts the x and y coordinates by specified values."""
    shift_x = shift_x.view(*xx.shape[:-2], 1, 1)
    shift_y = shift_y.view(*yy.shape[:-2], 1, 1)
    return xx - shift_x, yy - shift_y


def rotate(rotations, directions):
    s = directions.shape
    directions = directions.view(-1, 2, 1)
    rotations = rotations.view(-1, 2, 2)
    res = torch.bmm(rotations, directions).view(*s)
    return res


def flip_x(xx, yy):
    return torch.flip(xx, [-2]), torch.flip(yy, [-2])


class PolicyCostKMTaper(PolicyCostContinuous):
    """Cost with tapered end and using the kinematic model.
    The main difference is now we propagate through the kinematic model,
    and we can shift the car from the center of the image.
    """

    @dataclass
    class Config(PolicyCostContinuous.Config):
        masks_power_x: float = 2.0
        masks_power_y: float = 2.0
        agg_func_str: str = "sum"
        curl: int = 0
        rotate: bool = True
        mask_coeff: float = 10.0
        shifted_reference_frame: bool = False
        reference_distance_loss: bool = True
        # Destination lambda
        lambda_d: float = 0.0
        # Reference distance loss
        lambda_r: float = 1.0
        keep_dims: bool = False

    class Cost(NamedTuple):
        """Tuple to store the full result of the cost calculation."""

        state: PolicyCost.StateCosts
        uncertainty: torch.Tensor
        action: torch.Tensor
        jerk: torch.Tensor
        reference_distance: torch.Tensor
        total: torch.Tensor

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.agg_func = AggregationFunction(config.agg_func_str)

    def calculate_reference_distance_cost(
        self,
        scalar_states: torch.Tensor,
        reference_scalar_states: torch.Tensor,
    ):
        reference_distance_cost = 0.0
        if self.config.lambda_r > 0:
            diff_xy = UnitConverter.pixels_to_m(
                self.normalizer.unnormalize_states(scalar_states)[..., :2]
                - self.normalizer.unnormalize_states(reference_scalar_states)[
                    ..., :2
                ]
            ).abs()
            # Shapes are bsize, npred, 2 for each of them.
            reference_distance_cost = (
                torch.nn.functional.relu(diff_xy[:, :, 0] - 30) ** 4
                + torch.nn.functional.relu(diff_xy[:, :, 1] - 5) ** 4
            )
        return reference_distance_cost

    def get_masks(
        self,
        scalar_states: torch.Tensor,
        actions: torch.Tensor,  # needed for curling.
        context_state_seq: StateSequence,
        unnormalize: bool = False,
        *,
        metadata=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns two masks - one for proximity cost, so it has a flat nose,
        and one for lane and offroad costs, so it doesn't have a flat nose.
        """
        ref_states = context_state_seq.states

        (
            bsize,
            npred,
            nchannels,
            crop_h,
            crop_w,
        ) = context_state_seq.images.shape
        device = context_state_seq.images.device

        states = context_state_seq.states.view(bsize * npred, 5).clone()
        ref_states = ref_states.view(bsize * npred, 5).clone()

        if unnormalize:
            states = self.normalizer.unnormalize_states(states)
            ref_states = self.normalizer.unnormalize_states(ref_states)
            actions = self.normalizer.unnormalize_actions(actions)

        states = states.view(bsize, npred, 5)
        ref_states = ref_states.view(bsize, npred, 5)
        actions = actions.view(bsize, npred, 2)

        LANE_WIDTH_METRES = 3.7
        LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
        # SCALE = 1 / 4
        PIXELS_IN_METRE = LANE_WIDTH_PIXELS / LANE_WIDTH_METRES
        MAX_SPEED_MS = 130 / 3.6  # m/s
        LOOK_AHEAD_M = MAX_SPEED_MS  # meters
        LOOK_SIDEWAYS_M = 2 * LANE_WIDTH_METRES  # meters
        METRES_IN_FOOT = 0.3048
        TIMESTEP = 0.1

        car_size = context_state_seq.car_size.to(device)

        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * METRES_IN_FOOT
        width = width.view(bsize, 1)

        length = length * METRES_IN_FOOT
        length = length.view(bsize, 1)

        positions = UnitConverter.pixels_to_m(states[:, :, :2])
        ref_positions = UnitConverter.pixels_to_m(ref_states[:, :, :2])
        speeds_norm = states[:, :, 4] / PIXELS_IN_METRE
        speeds_norm_pixels = states[:, :, 4]

        alphas = torch.atan(speeds_norm_pixels * actions[:, :, 1] * TIMESTEP)
        gammas = (np.pi - alphas) / 2
        radii = (
            -1 * speeds_norm * TIMESTEP / (2 * torch.cos(gammas) + 1e-7)
        )  # in meters

        directions = states[:, :, 2:4]
        ref_directions = ref_states[:, :, 2:4]

        positions_adjusted = positions - ref_positions.detach()

        rotation = rotation_matrix(ref_directions.detach(), directions).to(
            device
        )

        y = torch.linspace(
            -LOOK_SIDEWAYS_M, LOOK_SIDEWAYS_M, crop_w, device=device
        )
        x = torch.linspace(-LOOK_AHEAD_M, LOOK_AHEAD_M, crop_h, device=device)
        xx, yy = torch.meshgrid(x, y)
        xx = xx.repeat(bsize, npred, 1, 1)
        yy = yy.repeat(bsize, npred, 1, 1)

        xx, yy = coordinate_shift(
            xx, yy, positions_adjusted[:, :, 0], positions_adjusted[:, :, 1]
        )
        if self.config.rotate > 0:
            xx, yy = coordinate_rotate_matrix(xx, yy, rotation)
        if self.config.curl > 0:
            xx, yy = coordinate_curl(xx, yy, radii)

        # Because originally x goes from negative to positive, and the
        # generated mask is overlayed with an image where the cars ahead of us
        # are positive distance, we flip x axis.
        xx, yy = flip_x(xx, yy)

        if metadata is not None:
            metadata["width_y"] = yy.abs() < (width / 2).view(bsize, 1, 1, 1)

        x_prime, y_prime = xx, yy

        REPEAT_SHAPE = (bsize, npred, 1, 1)
        # y_d - is lateral distance to 0 mask value - lateral safety distance
        y_d = width / 2 + LANE_WIDTH_METRES
        if metadata is not None:
            metadata["y_d"] = y_d
        # x_s - is longitudinal distance to 0 mask value - safety distance
        x_s = (
            1.5 * torch.clamp(speeds_norm.detach(), min=10) + length * 1.5 + 1
        )
        x_s = x_s.view(REPEAT_SHAPE)

        z_x_prime = torch.clamp(
            (x_s - torch.abs(x_prime))
            / (x_s - length.view(bsize, 1, 1, 1) / 2),
            min=0,
        )
        # z_x_prime_rotation = torch.clamp(
        #     (x_s_rotation - torch.abs(x_prime)) / (x_s_rotation), min=0
        # )
        if not z_x_prime.max() > 0:
            print("Seems like the car is outside the state image.")

        r_y_prime = torch.clamp(
            (y_d.view(bsize, 1, 1, 1) - torch.abs(y_prime))
            / (y_d - width / 2).view(bsize, 1, 1, 1),
            min=0,
        )
        if metadata is not None:
            metadata["r_y_prime_b"] = r_y_prime
            metadata["r_y_prime_b_width"] = r_y_prime > 1

        if not r_y_prime.max() > 0:
            print("Seems like the car is outside the state image.")
        r_y_prime_clamped = torch.clamp(r_y_prime, max=1)

        # Double max reduces the last two dimensions, leaving batch_size x
        # n_pred.
        # Min multiplier is 1 / max value of r_y_prime
        # to scale r_y to never be greater than 1?
        # min_multiplier = 1 / r_y_prime.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values

        # # What is this one below?
        # y_multiply_coefficient = torch.clamp(
        #     (2 * (min_multiplier - 1) * torch.abs(x_prime) + x_s - min_multiplier * length.view(bsize, 1, 1, 1))
        #     / (x_s - length.view(bsize, 1, 1, 1)),
        #     max=1,
        # )
        # # this clips it from the bottom to never go below min_multiplier.
        # y_multiply_coefficient = torch.max(y_multiply_coefficient, min_multiplier)
        # if metadata is not None:
        #     metadata['y_multiply_coefficient'] = y_multiply_coefficient

        # r_y_prime = r_y_prime * y_multiply_coefficient
        # r_y_prime_clamped = r_y_prime_clamped * y_multiply_coefficient

        r_y_prime = r_y_prime ** self.config.masks_power_y
        r_y_prime_clamped = r_y_prime_clamped ** self.config.masks_power_y

        # Acceleration probe
        x_major = z_x_prime ** self.config.masks_power_x
        # x_major[:, :, (x_major.shape[2] // 2 + 10) :, :] = (
        #     x_major[:, :, (x_major.shape[2] // 2 + 10) :, :].clone() ** 2
        # )

        if metadata is not None:
            metadata["x_major"] = x_major
            metadata["r_y_prime"] = r_y_prime
            metadata["r_y_prime_clamped"] = r_y_prime_clamped

        return x_major * r_y_prime_clamped, x_major * r_y_prime

    def get_traj_points(
        self,
        scalar_states: torch.Tensor,
        context_state_seq: StateSequence,
        unnormalize: bool = False,
    ):
        (
            bsize,
            npred,
            nchannels,
            crop_h,
            crop_w,
        ) = context_state_seq.images.shape
        device = scalar_states.device

        states = scalar_states.view(bsize * npred, 5).clone()
        ref_states = context_state_seq.states.view(bsize * npred, 5).clone()

        if unnormalize:
            states = self.normalizer.unnormalize_states(states)
            ref_states = self.normalizer.unnormalize_states(ref_states)

        states = states.view(bsize, npred, 5)
        ref_states = ref_states.view(bsize, npred, 5)

        LANE_WIDTH_METRES = 3.7
        # SCALE = 1 / 4
        MAX_SPEED_MS = 130 / 3.6  # m/s
        LOOK_AHEAD_M = MAX_SPEED_MS  # meters
        LOOK_SIDEWAYS_M = 2 * LANE_WIDTH_METRES  # meters
        METRES_IN_FOOT = 0.3048

        car_size = context_state_seq.car_size.to(device)

        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * METRES_IN_FOOT
        width = width.view(bsize, 1)

        length = length * METRES_IN_FOOT
        length = length.view(bsize, 1)

        positions = UnitConverter.pixels_to_m(states[:, :, :2])
        ref_positions = UnitConverter.pixels_to_m(ref_states[:, :, :2])

        directions = states[:, :, 2:4]
        ref_directions = ref_states[:, :, 2:4]

        positions_adjusted = positions - ref_positions.detach()

        rotation = rotation_matrix(ref_directions.detach(), directions).to(
            device
        )

        y = torch.linspace(
            -LOOK_SIDEWAYS_M, LOOK_SIDEWAYS_M, crop_w, device=device
        )
        x = torch.linspace(-LOOK_AHEAD_M, LOOK_AHEAD_M, crop_h, device=device)
        xx, yy = torch.meshgrid(x, y)
        xx = xx.repeat(bsize, npred, 1, 1)
        yy = yy.repeat(bsize, npred, 1, 1)

        xx, yy = coordinate_shift(
            xx, yy, positions_adjusted[:, :, 0], positions_adjusted[:, :, 1]
        )
        if self.config.rotate > 0:
            xx, yy = coordinate_rotate_matrix(xx, yy, rotation)

        # Because originally x goes from negative to positive, and the
        # generated mask is overlayed with an image where the cars ahead of us
        # are positive distance, we flip x axis.
        xx, yy = flip_x(xx, yy)
        x_prime, y_prime = xx, yy

        # draw a square of 1m
        z_x_prime = (x_prime.abs() < 0.5).int()
        z_y_prime = (y_prime.abs() < 0.5).int()

        return z_x_prime * z_y_prime

    def get_cost_landscape(self, image, masks, mask_sums):
        """
        Takes 1 image and 1 state and 1 action.
        Shifts the car around the image to build the cost landscape.
        """

        def shift(image, x, y):
            new_image = torch.roll(image, (x, y), (1, 2)).clone()
            if y >= 0:
                new_image[:, :, :y] = 0
            else:
                new_image[:, :, y:] = 0
            if x >= 0:
                new_image[:, :x, :] = 0
            else:
                new_image[:, x:, :] = 0
            return new_image

        results = torch.zeros(image.shape[1:])
        center_x = results.shape[0] // 2 + 1
        center_y = results.shape[1] // 2 + 1

        for x in range(results.shape[0]):
            for y in range(results.shape[1]):
                shift_x = center_x - x
                shift_y = center_y - y
                s_image = shift(image, shift_x, shift_y)
                p_cost = (
                    self.compute_proximity_cost_km(
                        s_image.unsqueeze(0).unsqueeze(0).cuda(),
                        masks,
                        mask_sums,
                    )
                    * self.config.mask_coeff
                    * self.config.lambda_p
                )
                lane_cost = (
                    self.compute_lane_cost_km(
                        s_image.unsqueeze(0).unsqueeze(0).cuda(),
                        masks,
                        mask_sums,
                    )
                    * self.config.mask_coeff
                    * self.config.lambda_l
                )
                offroad_cost = (
                    self.compute_offroad_cost_km(
                        s_image.unsqueeze(0).unsqueeze(0).cuda(),
                        masks,
                        mask_sums,
                    )
                    * self.config.mask_coeff
                    * self.config.lambda_o
                )

                results[x][y] = p_cost + lane_cost + offroad_cost
        return results

    def compute_state_costs_for_training(
        # self, inputs, pred_images, pred_states, pred_actions, car_sizes
        self,
        scalar_states: torch.Tensor,
        actions: torch.Tensor,
        context_state_seq: StateSequence,
        conditional_state_seq: Optional[StateSequence] = None,
    ):
        """Costs associated with masks are state costs"""
        proximity_mask, proximity_mask_lo = self.get_masks(
            scalar_states,
            actions,
            context_state_seq,
            unnormalize=False,
        )
        mask_sums = proximity_mask.sum(dim=(-1, -2))
        mask_sums_lo = proximity_mask_lo.sum(dim=(-1, -2))

        # We impose a cost for being too far from the reference. Being too far to the side is punished more.
        proximity_cost = self.compute_proximity_cost_km(
            context_state_seq.images, proximity_mask, mask_sums
        )
        lane_cost = self.compute_lane_cost_km(
            context_state_seq.images, proximity_mask_lo, mask_sums_lo
        )
        offroad_cost = self.compute_offroad_cost_km(
            context_state_seq.images, proximity_mask_lo, mask_sums_lo
        )

        batch_size = lane_cost.shape[0]
        # Multiply everything with mask_coeff used to scale up the costs that
        # depend on mask size.
        lane_total = (
            self.apply_gamma(lane_cost).view(batch_size, -1).mean(dim=-1)
        )
        offroad_total = (
            self.apply_gamma(offroad_cost).view(batch_size, -1).mean(dim=-1)
        )
        proximity_total = (
            self.apply_gamma(proximity_cost).view(batch_size, -1).mean(dim=-1)
        )

        return PolicyCost.StateCosts(
            total_proximity=proximity_total,
            total_lane=lane_total,
            total_offroad=offroad_total,
            proximity=proximity_cost,
            offroad=offroad_cost,
            lane=lane_cost,
        )

    def compute_combined_loss(
        self,
        state: PolicyCost.StateCosts,
        uncertainty: torch.Tensor,
        action: torch.Tensor,
        jerk: torch.Tensor,
        reference_distance: torch.Tensor,
    ):
        return (
            self.config.lambda_p * state.total_proximity
            + self.config.lambda_o * state.total_offroad
            + self.config.lambda_l * state.total_lane
            + self.config.u_reg * uncertainty
            + self.config.lambda_a * action
            + self.config.lambda_j * jerk
            + self.config.lambda_r * reference_distance
        )

    def compute_proximity_cost_km(
        self, images, proximity_masks, masks_sums=None
    ):
        images = images[:, :, 1]
        if self.config.skip_contours:
            images = self.compute_contours(images)
        images = images ** 2
        return self._multiply_masks_km(images, proximity_masks, masks_sums)

    def compute_lane_cost_km(self, images, proximity_masks, masks_sums=None):
        images = images[:, :, 0] ** 2
        return self._multiply_masks_km(images, proximity_masks, masks_sums)

    def compute_offroad_cost_km(
        self, images, proximity_masks, masks_sums=None
    ):
        images = images[:, :, 2] ** 2
        return self._multiply_masks_km(images, proximity_masks, masks_sums)

    def _multiply_masks_km(
        self,
        images: torch.Tensor,
        proximity_mask: torch.Tensor,
        proximity_mask_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Given an image of shape bsize, npred, height, width and a mask
        of same shape, combines them with the specified aggregation function.
        Returns tensor of shape bsize, npred.
        """
        bsize, npred = images.shape[:2]
        costs = proximity_mask * images
        if proximity_mask_sum is not None:
            costs /= proximity_mask_sum.view(*proximity_mask_sum.shape, 1, 1)
        costs_m = self.agg_func(costs.view(bsize, npred, -1), 2)
        return costs_m.view(bsize, npred)

    def calculate_cost(
        self,
        conditional_state_seq: StateSequence,
        scalar_states: torch.Tensor,
        actions: torch.Tensor,
        context_state_seq: StateSequence,
    ) -> Cost:
        batch_size = actions.shape[0]
        u_loss = self.calculate_uncertainty_cost(
            conditional_state_seq, actions
        )
        loss_a = self.calculate_action_loss(actions)
        loss_j = self.calculate_jerk_loss(actions)

        reference_distance_cost = self.calculate_reference_distance_cost(
            scalar_states, context_state_seq.states
        )
        reference_total = (
            self.apply_gamma(reference_distance_cost)
            .view(batch_size, -1)
            .mean(dim=-1)
        )

        state_cost = self.compute_state_costs_for_training(
            scalar_states,
            actions,
            context_state_seq,
        )

        total = self.compute_combined_loss(
            state=state_cost,
            uncertainty=u_loss,
            action=loss_a,
            jerk=loss_j,
            reference_distance=reference_total,
        )

        return PolicyCostKMTaper.Cost(
            state=state_cost,
            uncertainty=u_loss,
            action=loss_a,
            jerk=loss_j,
            total=total,
            reference_distance=reference_total,
        )

    def _build_cost_landscape(
        self,
        scalar_states: torch.Tensor,
        context_state_seq: torch.Tensor,
        conditional_state_seq: torch.Tensor,
        proximity_mask: torch.Tensor,
        proximity_mask_sum: torch.Tensor,
    ):
        self.overlay = (
            proximity_mask.unsqueeze(2) * 0.85 + context_state_seq.images
        )

        # this is to only update image every 10 times.
        if not hasattr(self, "ctr"):
            self.ctr = 0
            self.t_image = torch.zeros(10, 10)
            self.t_image_data = None

        if hasattr(self, "traj_landscape") and self.traj_landscape:
            if self.ctr % 10 == 0:
                last_image = conditional_state_seq.images[:, -1:, :3].repeat(
                    1, context_state_seq.states.shape[1], 1, 1, 1
                )
                last_state = conditional_state_seq.states.repeat(
                    1, context_state_seq.states.shape[1], 1
                )
                new_context = StateSequence(
                    images=last_image,
                    states=last_state,
                    car_sizes=context_state_seq.car_sizes,
                    ego_car_image=conditional_state_seq.ego_car_image,
                )

                traj_proximity_mask = self.get_traj_points(
                    scalar_states,
                    new_context,
                    unnormalize=False,
                )

                cost_landscape_unnormed = self.get_cost_landscape(
                    last_image[0, 0].cuda(),
                    proximity_mask[0, 0].unsqueeze(0).unsqueeze(0),
                    proximity_mask_sum[0, 0].unsqueeze(0).unsqueeze(0),
                )
                cost_landscape = cost_landscape_unnormed / (
                    cost_landscape_unnormed.max() + 1e-9
                )
                cost_landscape = torch.stack(
                    [
                        torch.zeros_like(cost_landscape),
                        cost_landscape,
                        torch.zeros_like(cost_landscape),
                    ]
                )

                self.t_image = (
                    cost_landscape.cuda() + traj_proximity_mask.sum(dim=1)[0]
                )
                self.t_image = self.t_image.contiguous()
                self.t_image_data = (
                    cost_landscape_unnormed.detach().cpu().numpy()
                )

            self.ctr += 1

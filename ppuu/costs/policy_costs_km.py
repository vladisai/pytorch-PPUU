"""Cost model that uses the kinematic model.
"""

from dataclasses import dataclass, field

import torch
import numpy as np

from ppuu.costs.policy_costs_continuous import PolicyCostContinuous


class AggregationFunction:
    def __init__(self, s):
        self.s = s

    def __call__(self, *args, **kwargs):
        if self.s == "sum":
            return torch.sum(*args, **kwargs)
        elif self.s.startswith("logsumexp"):
            beta = float(self.s.split("-")[1])
            return 1 / beta * torch.logsumexp(args[0] * beta, *args[1:])
        else:
            return torch.logsumexp(*args, **kwargs)


class PolicyCostKM(PolicyCostContinuous):
    @dataclass
    class Config(PolicyCostContinuous.Config):
        masks_power_x: int = 4.22
        masks_power_y: int = 4.22
        agg_func_str: str = "logsumexp-67"
        lambda_a: float = field(default=0.02)
        lambda_l: float = field(default=1.77)
        lambda_o: float = field(default=0.5)
        lambda_p: float = field(default=2.15)

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.agg_func = AggregationFunction(config.agg_func_str)

    def get_masks(self, images, states, car_size, unnormalize):
        bsize, npred, nchannels, crop_h, crop_w = images.shape
        device = images.device

        states = states.view(bsize * npred, 4).clone()

        if unnormalize:
            states = states * (
                1e-8
                + self.data_stats["s_std"].view(1, 5).expand(states.size())
            ).to(device)
            states = states + self.data_stats["s_mean"].view(1, 5).expand(
                states.size()
            ).to(device)

        states = states.view(bsize, npred, 5)

        LANE_WIDTH_METRES = 3.7
        LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
        # SCALE = 1 / 4
        PIXELS_IN_METRE = LANE_WIDTH_PIXELS / LANE_WIDTH_METRES
        MAX_SPEED_MS = 130 / 3.6  # m/s
        LOOK_AHEAD_M = MAX_SPEED_MS  # meters
        LOOK_SIDEWAYS_M = 2 * LANE_WIDTH_METRES  # meters
        METRES_IN_FOOT = 0.3048

        car_size = car_size.to(device)
        positions = states[:, :, :2]
        speeds_norm = states[:, :, 4] / PIXELS_IN_METRE
        # speeds_o = torch.ones_like(speeds).cuda()
        # directions = torch.atan2(speeds_o[:, :, 1], speeds_o[:, :, 0])
        directions = states[:, :, 2:4]

        positions_adjusted = positions - positions.detach()
        # here we flip directions because they're flipped otherwise
        directions_adjusted = -(directions - directions.detach())

        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * METRES_IN_FOOT
        width = width.view(bsize, 1)

        length = length * METRES_IN_FOOT
        length = length.view(bsize, 1)

        REPEAT_SHAPE = (bsize, npred, 1, 1)

        y_d = width / 2 + LANE_WIDTH_METRES
        x_s = (
            self.config.safe_factor * torch.clamp(speeds_norm.detach(), min=10)
            + length * 1.5
            + 1
        )

        x_s = x_s.view(REPEAT_SHAPE)
        # x_s_rotation = torch.ones(REPEAT_SHAPE).cuda() * 1

        y = torch.linspace(
            -LOOK_SIDEWAYS_M, LOOK_SIDEWAYS_M, crop_w, device=device
        )
        # x should be from positive to negative, as when we draw the image,
        # cars with positive distance are ahead of us.
        # also, if it's reversed, the -x_pos in x_prime calculation becomes
        # +x_pos.
        x = torch.linspace(LOOK_AHEAD_M, -LOOK_AHEAD_M, crop_h, device=device)
        xx, yy = torch.meshgrid(x, y)
        xx = xx.repeat(bsize, npred, 1, 1)
        yy = yy.repeat(bsize, npred, 1, 1)

        c, s = torch.cos(directions_adjusted), torch.sin(directions_adjusted)
        c = c.view(REPEAT_SHAPE)
        s = s.view(REPEAT_SHAPE)

        x_pos = positions_adjusted[:, :, 0]
        x_pos = x_pos.view(*x_pos.shape, 1, 1)
        y_pos = positions_adjusted[:, :, 1]
        y_pos = y_pos.view(*y_pos.shape, 1, 1)

        x_prime = c * xx - s * yy - x_pos  # <- here
        y_prime = s * xx + c * yy - y_pos  # and here a double - => +

        z_x_prime = torch.clamp(
            (x_s - torch.abs(x_prime))
            / (x_s - length.view(bsize, 1, 1, 1) / 2),
            min=0,
        )
        # z_x_prime_rotation = torch.clamp(
        #     (x_s_rotation - torch.abs(x_prime)) / (x_s_rotation), min=0
        # )
        r_y_prime = torch.clamp(
            (y_d.view(bsize, 1, 1, 1) - torch.abs(y_prime))
            / (y_d - width / 2).view(bsize, 1, 1, 1),
            min=0,
        )

        # Acceleration probe
        x_major = z_x_prime ** self.config.masks_power_x
        # x_major[:, :, (x_major.shape[2] // 2 + 10):, :] =
        # = x_major[:, :, (x_major.shape[2] // 2 + 10):, :].clone() ** 2
        y_ramp = torch.clamp(r_y_prime ** self.config.masks_power_y, max=1)
        result_acceleration = x_major * y_ramp

        # Rotation probe
        # x_ramp = torch.clamp(z_x_prime, max=1).float()
        x_ramp = torch.clamp(z_x_prime ** self.config.masks_power_x, max=1)
        # x_ramp = (z_x_prime > 0).float()
        # x_ramp[:, :, (x_ramp.shape[2] // 2 + 10):, :]
        # = x_ramp[:, :, (x_ramp.shape[2] // 2 + 10):, :].clone() ** 2
        y_major = r_y_prime ** self.config.masks_power_y
        result_rotation = x_ramp * y_major

        return result_rotation, result_acceleration

    def compute_proximity_cost_km(self, images, proximity_masks):
        bsize, npred, nchannels, crop_h, crop_w = images.shape
        images = images.view(-1, nchannels, crop_h, crop_w)
        green_contours = self.compute_contours(images)
        green_contours = green_contours.view(bsize, npred, crop_h, crop_w)
        pre_max = proximity_masks * (green_contours ** 2)
        costs = self.agg_func(pre_max.view(bsize, npred, -1), 2)

        result = {}
        result["costs"] = costs
        result["masks"] = proximity_masks
        result["pre_max"] = pre_max
        result["contours"] = green_contours
        return result

    def compute_lane_cost_km(self, images, proximity_masks):
        bsize, npred = images.shape[0], images.shape[1]
        lanes = images[:, :, 0].float()
        costs = proximity_masks * (lanes ** 2)
        costs_m = self.agg_func(costs.view(bsize, npred, -1), 2)
        return costs_m.view(bsize, npred)

    def compute_offroad_cost_km(self, images, proximity_masks):
        bsize, npred = images.shape[0], images.shape[1]
        offroad = images[:, :, 2]
        costs = proximity_masks * (offroad ** 2)
        costs_m = self.agg_func(costs.view(bsize, npred, -1), 2)
        return costs_m.view(bsize, npred)

    def compute_state_costs_for_training(
        self, pred_images, pred_states, pred_actions, car_sizes
    ):
        proximity_masks = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states["km"],
            car_sizes,
            unnormalize=True,
        )

        npred = pred_images.size(1)
        gamma_mask = (
            torch.tensor([0.99 ** t for t in range(npred + 1)])
            .cuda()
            .unsqueeze(0)
        )
        proximity_cost = self.compute_proximity_cost_km(
            pred_images, proximity_masks[1],
        )["costs"]

        lane_cost = self.compute_lane_cost_km(pred_images, proximity_masks[0])
        offroad_cost = self.compute_offroad_cost_km(
            pred_images, proximity_masks[0]
        )

        lane_loss = torch.mean(lane_cost * gamma_mask[:, :npred])
        offroad_loss = torch.mean(offroad_cost * gamma_mask[:, :npred])
        proximity_loss = torch.mean(proximity_cost * gamma_mask[:, :npred])
        return dict(
            proximity_cost=proximity_cost,
            lane_cost=lane_cost,
            offroad_cost=offroad_cost,
            lane_loss=lane_loss,
            offroad_loss=offroad_loss,
            proximity_loss=proximity_loss,
        )


class PolicyCostKMSplit(PolicyCostKM):
    def compute_state_costs_for_training(
        self, pred_images, pred_states, pred_actions, car_sizes
    ):
        proximity_mask_a = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states["km_a"],
            car_sizes,
            unnormalize=True,
        )[1]
        proximity_mask_b = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states["km_b"],
            car_sizes,
            unnormalize=True,
        )[0]

        npred = pred_images.size(1)
        gamma_mask = (
            torch.tensor([0.99 ** t for t in range(npred + 1)])
            .cuda()
            .unsqueeze(0)
        )
        proximity_cost = (
            self.compute_proximity_cost_km(pred_images, proximity_mask_a,)[
                "costs"
            ]
            + self.compute_proximity_cost_km(pred_images, proximity_mask_b,)[
                "costs"
            ]
        ) / 2
        lane_cost = (
            self.compute_lane_cost_km(pred_images, proximity_mask_a)
            + self.compute_lane_cost_km(pred_images, proximity_mask_b)
        ) / 2
        offroad_cost = (
            self.compute_offroad_cost_km(pred_images, proximity_mask_a)
            + self.compute_offroad_cost_km(pred_images, proximity_mask_b)
        ) / 2

        lane_loss = torch.mean(lane_cost * gamma_mask[:, :npred])
        offroad_loss = torch.mean(offroad_cost * gamma_mask[:, :npred])
        proximity_loss = torch.mean(proximity_cost * gamma_mask[:, :npred])
        return dict(
            proximity_cost=proximity_cost,
            lane_cost=lane_cost,
            offroad_cost=offroad_cost,
            lane_loss=lane_loss,
            offroad_loss=offroad_loss,
            proximity_loss=proximity_loss,
        )


class PolicyCostKMTaper(PolicyCostKM):
    """Cost with tapered end"""

    @dataclass
    class Config(PolicyCostContinuous.Config):
        masks_power_x: int = 4.22
        masks_power_y: int = 4.22
        agg_func_str: str = "sum"
        lambda_a: float = field(default=0.23)
        lambda_j: float = field(default=2.5)
        lambda_l: float = field(default=2.86)
        lambda_o: float = field(default=3.18)
        lambda_p: float = field(default=25.5)
        u_reg: float = field(default=1.62)

    def get_masks(self, images, states, actions, car_size, unnormalize):
        bsize, npred, nchannels, crop_h, crop_w = images.shape
        device = images.device

        states = states.view(bsize * npred, 4).clone()

        if unnormalize:
            states = states * (
                1e-8
                + self.data_stats["s_std"].view(1, 5).expand(states.size())
            ).to(device)
            states = states + self.data_stats["s_mean"].view(1, 5).expand(
                states.size()
            ).to(device)

            actions = actions * (
                1e-8
                + self.data_stats["a_std"].view(1, 2).expand(actions.size())
            ).to(device)
            actions = actions + self.data_stats["a_mean"].view(1, 2).expand(
                actions.size()
            ).to(device)

        states = states.view(bsize, npred, 5)
        actions = actions.view(bsize, npred, 2)
        clipped_actions = actions[:, 1:, :]
        last_action = actions[:, -1, :].unsqueeze(1)
        stitched_actions = torch.cat((clipped_actions, last_action), dim=1)  #

        LANE_WIDTH_METRES = 3.7
        LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
        # SCALE = 1 / 4
        PIXELS_IN_METRE = LANE_WIDTH_PIXELS / LANE_WIDTH_METRES
        MAX_SPEED_MS = 130 / 3.6  # m/s
        LOOK_AHEAD_M = MAX_SPEED_MS  # meters
        LOOK_SIDEWAYS_M = 2 * LANE_WIDTH_METRES  # meters
        METRES_IN_FOOT = 0.3048
        TIMESTEP = 0.1

        car_size = car_size.to(device)

        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * METRES_IN_FOOT
        width = width.view(bsize, 1)

        length = length * METRES_IN_FOOT
        length = length.view(bsize, 1)

        positions = states[:, :, :2]
        speeds_norm = states[:, :, 4] / PIXELS_IN_METRE
        speeds_norm_pixels = states[:, :, 4]

        alphas = torch.atan(speeds_norm_pixels * actions[:, :, 1] * TIMESTEP)
        gammas = (np.pi - alphas) / 2
        radii = (
            -1 * speeds_norm * TIMESTEP / (2 * torch.cos(gammas) + 1e-7)
        )  # in meters

        # speeds_o = torch.ones_like(speeds).cuda()
        # directions = torch.atan2(speeds_o[:, :, 1], speeds_o[:, :, 0])
        directions = states[:, :, 2:4]

        positions_adjusted = positions - positions.detach()
        # here we flip directions because they're flipped otherwise
        directions_adjusted = -(directions - directions.detach())

        REPEAT_SHAPE = (bsize, npred, 1, 1)

        y_d = width / 2 + LANE_WIDTH_METRES
        x_s = (
            1.5 * torch.clamp(speeds_norm.detach(), min=10) + length * 1.5 + 1
        )

        x_s = x_s.view(REPEAT_SHAPE)
        # x_s_rotation = torch.ones(REPEAT_SHAPE).cuda() * 1

        y = torch.linspace(
            -LOOK_SIDEWAYS_M, LOOK_SIDEWAYS_M, crop_w, device=device
        )
        # x should be from positive to negative, as when we draw the image,
        # cars with positive distance are ahead of us.
        # also, if it's reversed, the -x_pos in x_prime calculation becomes
        # +x_pos.
        x = torch.linspace(LOOK_AHEAD_M, -LOOK_AHEAD_M, crop_h, device=device)
        xx, yy = torch.meshgrid(x, y)
        xx = xx.repeat(bsize, npred, 1, 1)
        yy = yy.repeat(bsize, npred, 1, 1)

        center_y = radii.unsqueeze(-1).unsqueeze(-1)
        center_x = torch.zeros_like(center_y)

        r = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        x1 = xx - center_x

        is_to_the_right = (center_y > 0).float() * 2 - 1
        y1 = (center_y - yy) * is_to_the_right

        alpha = torch.atan2(x1, y1)
        ll = alpha * r
        yy = center_y - is_to_the_right * r
        xx = center_x + ll

        c, s = torch.cos(directions_adjusted), torch.sin(directions_adjusted)
        c = c.view(REPEAT_SHAPE)
        s = s.view(REPEAT_SHAPE)

        x_pos = positions_adjusted[:, :, 0]
        x_pos = x_pos.view(*x_pos.shape, 1, 1)
        y_pos = positions_adjusted[:, :, 1]
        y_pos = y_pos.view(*y_pos.shape, 1, 1)

        x_prime = c * xx - s * yy - x_pos  # <- here
        y_prime = s * xx + c * yy - y_pos  # and here a double - => +

        z_x_prime = torch.clamp(
            (x_s - torch.abs(x_prime))
            / (x_s - length.view(bsize, 1, 1, 1) / 2),
            min=0,
        )
        # z_x_prime_rotation = torch.clamp(
        #     (x_s_rotation - torch.abs(x_prime)) / (x_s_rotation), min=0
        # )
        r_y_prime = torch.clamp(
            (y_d.view(bsize, 1, 1, 1) - torch.abs(y_prime))
            / (y_d - width / 2).view(bsize, 1, 1, 1),
            min=0,
        )

        # Double max reduces the last two dimensions, leaving batch_size x
        # n_pred.
        min_multiplier = (
            1
            / r_y_prime.max(dim=-1, keepdim=True)
            .values.max(dim=-2, keepdim=True)
            .values
        )
        y_multiply_coefficient = torch.clamp(
            (
                2 * (min_multiplier - 1) * torch.abs(x_prime)
                + x_s
                - min_multiplier * length.view(bsize, 1, 1, 1)
            )
            / (x_s - length.view(bsize, 1, 1, 1)),
            max=1,
        )
        # this clips it from the bottom to never go below min_multiplier.
        y_multiply_coefficient = torch.max(
            y_multiply_coefficient, min_multiplier
        )
        r_y_prime = torch.clamp(r_y_prime * y_multiply_coefficient, max=1)
        r_y_prime = r_y_prime ** self.config.masks_power_y

        # Acceleration probe
        x_major = z_x_prime ** self.config.masks_power_x
        # x_major[:, :, (x_major.shape[2] // 2 + 10) :, :] = (
        #     x_major[:, :, (x_major.shape[2] // 2 + 10) :, :].clone() ** 2
        # )
        return x_major * r_y_prime  # , x_major, r_y_prime

    def compute_state_costs_for_training(
        self, pred_images, pred_states, pred_actions, car_sizes
    ):
        device = pred_images.device
        proximity_mask = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states["km"],
            pred_actions,
            car_sizes,
            unnormalize=True,
        )

        npred = pred_images.size(1)
        gamma_mask = (
            torch.tensor([0.99 ** t for t in range(npred + 1)])
            .to(device)
            .unsqueeze(0)
        )
        proximity_cost = self.compute_proximity_cost_km(
            pred_images, proximity_mask,
        )["costs"]

        lane_cost = self.compute_lane_cost_km(pred_images, proximity_mask)
        offroad_cost = self.compute_offroad_cost_km(
            pred_images, proximity_mask
        )

        lane_loss = torch.mean(lane_cost * gamma_mask[:, :npred])
        offroad_loss = torch.mean(offroad_cost * gamma_mask[:, :npred])
        proximity_loss = torch.mean(proximity_cost * gamma_mask[:, :npred])
        return dict(
            proximity_cost=proximity_cost,
            lane_cost=lane_cost,
            offroad_cost=offroad_cost,
            lane_loss=lane_loss,
            offroad_loss=offroad_loss,
            proximity_loss=proximity_loss,
        )

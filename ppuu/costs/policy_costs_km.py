"""Cost model that uses the kinematic model.
"""

from dataclasses import dataclass, field

import torch
import numpy as np

from ppuu.costs.policy_costs_continuous import PolicyCostContinuous
from ppuu.data.dataloader import UnitConverter


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
        curl: bool = field(default=False)
        keep_dims: bool = False

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.agg_func = AggregationFunction(config.agg_func_str)

    # def get_masks(self, images, states, car_size, unnormalize):
    #     bsize, npred, nchannels, crop_h, crop_w = images.shape
    #     device = images.device

    #     states = states.view(bsize * npred, 4).clone()

    #     if unnormalize:
    #         states = self.normalizer.unnormalize_states(states)

    #     states = states.view(bsize, npred, 5)

    #     LANE_WIDTH_METRES = 3.7
    #     LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
    #     # SCALE = 1 / 4
    #     PIXELS_IN_METRE = LANE_WIDTH_PIXELS / LANE_WIDTH_METRES
    #     MAX_SPEED_MS = 130 / 3.6  # m/s
    #     LOOK_AHEAD_M = MAX_SPEED_MS  # meters
    #     LOOK_SIDEWAYS_M = 2 * LANE_WIDTH_METRES  # meters
    #     METRES_IN_FOOT = 0.3048

    #     car_size = car_size.to(device)
    #     positions = states[:, :, :2]
    #     speeds_norm = states[:, :, 4] / PIXELS_IN_METRE
    #     # speeds_o = torch.ones_like(speeds).cuda()
    #     old_directions = directions
    #     directions = torch.atan2(directions[:, :, 3], directions[:, :, 2])
    #     # directions = states[:, :, 2:4]

    #     positions_adjusted = positions - positions.detach()
    #     # here we flip directions because they're flipped otherwise
    #     directions_adjusted = -(directions - directions.detach())

    #     width, length = car_size[:, 0], car_size[:, 1]  # feet
    #     width = width * METRES_IN_FOOT
    #     width = width.view(bsize, 1)

    #     length = length * METRES_IN_FOOT
    #     length = length.view(bsize, 1)

    #     REPEAT_SHAPE = (bsize, npred, 1, 1)

    #     y_d = width / 2 + LANE_WIDTH_METRES
    #     x_s = self.config.safe_factor * torch.clamp(speeds_norm.detach(), min=10) + length * 1.5 + 1

    #     x_s = x_s.view(REPEAT_SHAPE)
    #     # x_s_rotation = torch.ones(REPEAT_SHAPE).cuda() * 1

    #     y = torch.linspace(-LOOK_SIDEWAYS_M, LOOK_SIDEWAYS_M, crop_w, device=device)
    #     # x should be from positive to negative, as when we draw the image,
    #     # cars with positive distance are ahead of us.
    #     # also, if it's reversed, the -x_pos in x_prime calculation becomes
    #     # +x_pos.
    #     x = torch.linspace(LOOK_AHEAD_M, -LOOK_AHEAD_M, crop_h, device=device)
    #     xx, yy = torch.meshgrid(x, y)
    #     xx = xx.repeat(bsize, npred, 1, 1)
    #     yy = yy.repeat(bsize, npred, 1, 1)

    #     c, s = torch.cos(directions_adjusted), torch.sin(directions_adjusted)
    #     c = c.view(REPEAT_SHAPE)
    #     s = s.view(REPEAT_SHAPE)

    #     x_pos = positions_adjusted[:, :, 0]
    #     x_pos = x_pos.view(*x_pos.shape, 1, 1)
    #     y_pos = positions_adjusted[:, :, 1]
    #     y_pos = y_pos.view(*y_pos.shape, 1, 1)

    #     x_prime = c * xx - s * yy - x_pos  # <- here
    #     y_prime = s * xx + c * yy - y_pos  # and here a double - => +

    #     z_x_prime = torch.clamp((x_s - torch.abs(x_prime)) / (x_s - length.view(bsize, 1, 1, 1) / 2), min=0,)
    #     # z_x_prime_rotation = torch.clamp(
    #     #     (x_s_rotation - torch.abs(x_prime)) / (x_s_rotation), min=0
    #     # )
    #     r_y_prime = torch.clamp(
    #         (y_d.view(bsize, 1, 1, 1) - torch.abs(y_prime)) / (y_d - width / 2).view(bsize, 1, 1, 1), min=0,
    #     )

    #     # Acceleration probe
    #     x_major = z_x_prime ** self.config.masks_power_x
    #     # x_major[:, :, (x_major.shape[2] // 2 + 10):, :] =
    #     # = x_major[:, :, (x_major.shape[2] // 2 + 10):, :].clone() ** 2
    #     y_ramp = torch.clamp(r_y_prime ** self.config.masks_power_y, max=1)
    #     result_acceleration = x_major * y_ramp

    #     # Rotation probe
    #     # x_ramp = torch.clamp(z_x_prime, max=1).float()
    #     x_ramp = torch.clamp(z_x_prime ** self.config.masks_power_x, max=1)
    #     # x_ramp = (z_x_prime > 0).float()
    #     # x_ramp[:, :, (x_ramp.shape[2] // 2 + 10):, :]
    #     # = x_ramp[:, :, (x_ramp.shape[2] // 2 + 10):, :].clone() ** 2
    #     y_major = r_y_prime ** self.config.masks_power_y
    #     result_rotation = x_ramp * y_major

    #     return result_rotation, result_acceleration

    def compute_proximity_cost_km(
        self, images, proximity_masks, masks_sums=None
    ):
        bsize, npred, nchannels, crop_h, crop_w = images.shape
        images = images.view(-1, nchannels, crop_h, crop_w)
        green_contours = self.compute_contours(images)
        green_contours = green_contours.view(bsize, npred, crop_h, crop_w)
        pre_max = proximity_masks * (green_contours ** 2)
        if masks_sums is not None:
            pre_max /= masks_sums.view(*masks_sums.shape, 1, 1)
        costs = self.agg_func(pre_max.view(bsize, npred, -1), 2)
        result = {}
        result["costs"] = costs
        result["masks"] = proximity_masks
        result["pre_max"] = pre_max
        result["contours"] = green_contours

        return result

    def compute_lane_cost_km(self, images, proximity_masks, masks_sums=None):
        bsize, npred = images.shape[0], images.shape[1]
        lanes = images[:, :, 0].float()
        costs = proximity_masks * (lanes ** 2)
        if masks_sums is not None:
            costs /= masks_sums.view(*masks_sums.shape, 1, 1)
        costs_m = self.agg_func(costs.view(bsize, npred, -1), 2)
        return costs_m.view(bsize, npred)

    def compute_offroad_cost_km(
        self, images, proximity_masks, masks_sums=None
    ):
        bsize, npred = images.shape[0], images.shape[1]
        offroad = images[:, :, 2]
        costs = proximity_masks * (offroad ** 2)
        if masks_sums is not None:
            costs /= masks_sums.view(*masks_sums.shape, 1, 1)
        costs_m = self.agg_func(costs.view(bsize, npred, -1), 2)
        return costs_m.view(bsize, npred)

    def compute_combined_loss(
        self,
        proximity_loss,
        uncertainty_loss,
        lane_loss,
        action_loss,
        jerk_loss,
        offroad_loss,
        speed_loss,
        reference_distance_loss,
        **_kwargs,
    ):
        return (
            self.config.lambda_p * proximity_loss
            + self.config.u_reg * uncertainty_loss
            + self.config.lambda_l * lane_loss
            + self.config.lambda_a * action_loss
            + self.config.lambda_o * offroad_loss
            + self.config.lambda_j * jerk_loss
            + self.config.lambda_s * speed_loss
            + self.config.lambda_r * reference_distance_loss
        )

    def compute_state_costs_for_training(
        self, _, pred_images, pred_states, _pred_actions, car_sizes
    ):
        proximity_masks = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states,
            car_sizes,
            unnormalize=True,
        )

        npred = pred_images.size(1)
        gamma_mask = (
            torch.tensor([self.config.gamma ** t for t in range(npred + 1)])
            .cuda()
            .unsqueeze(0)
        )
        proximity_cost = self.compute_proximity_cost_km(
            pred_images,
            proximity_masks[1],
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


def coordinate_curl(xx, yy, radii):
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


def coordinate_rotate(xx, yy, dx, dy):
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


def coordinate_shift(xx, yy, shift_x, shift_y):
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


class PolicyCostKMTaper(PolicyCostKM):
    """Cost with tapered end"""

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
        lambda_s: float = field(default=0.0)
        target_speed: float = field(default=80)
        # lambda_a: float = field(default=0.23)
        # lambda_j: float = field(default=2.5)
        # lambda_l: float = field(default=2.86)
        # lambda_o: float = field(default=3.18)
        # lambda_p: float = field(default=45.5)
        # u_reg: float = field(default=1.62)

    def get_masks(
        self,
        images,
        states,
        actions,
        car_size,
        unnormalize,
        ref_states=None,
        *,
        metadata=None,
    ):
        """Returns two masks - one for proximity cost, so it has a flat nose,
        and one for lane and offroad costs, so it doesn't have a flat nose.
        """
        if ref_states is None:
            ref_states = states

        bsize, npred, nchannels, crop_h, crop_w = images.shape
        device = images.device

        states = states.view(bsize * npred, 5).clone()
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

        car_size = car_size.to(device)

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

        rotation = rotation_matrix(directions, ref_directions.detach()).to(
            device
        )

        y = torch.linspace(
            -LOOK_SIDEWAYS_M, LOOK_SIDEWAYS_M, crop_w, device=device
        )
        x = torch.linspace(LOOK_AHEAD_M, -LOOK_AHEAD_M, crop_h, device=device)
        xx, yy = torch.meshgrid(x, y)
        xx = xx.repeat(bsize, npred, 1, 1)
        yy = yy.repeat(bsize, npred, 1, 1)

        # breakpoint()

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
        # xx, yy = flip_x(xx, yy)

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
        self, images, states, actions, car_size, unnormalize, ref_states=None
    ):
        if ref_states is None:
            ref_states = states

        bsize, npred, nchannels, crop_h, crop_w = images.shape
        device = images.device

        states = states.view(bsize * npred, 5).clone()
        ref_states = ref_states.view(bsize * npred, 5).clone()

        if unnormalize:
            states = self.normalizer.unnormalize_states(states)
            ref_states = self.normalizer.unnormalize_states(ref_states)
            actions = self.normalizer.unnormalize_actions(actions)

        states = states.view(bsize, npred, 5)
        ref_states = ref_states.view(bsize, npred, 5)
        actions = actions.view(bsize, npred, 2)

        LANE_WIDTH_METRES = 3.7
        # SCALE = 1 / 4
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

        positions = UnitConverter.pixels_to_m(states[:, :, :2])
        ref_positions = UnitConverter.pixels_to_m(ref_states[:, :, :2])
        speeds_norm = UnitConverter.pixels_to_m(states[:, :, 4])
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
                # p_cost = cost.compute_proximity_cost(s_image.unsqueeze(0).unsqueeze(0), states.unsqueeze(0), car_sizes.unsqueeze(0), unnormalize=True)
                p_cost = (
                    self.compute_proximity_cost_km(
                        s_image.unsqueeze(0).unsqueeze(0).cuda(),
                        masks,
                        mask_sums,
                    )["costs"]
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
        self, inputs, pred_images, pred_states, pred_actions, car_sizes
    ):
        device = pred_images.device

        if self.config.shifted_reference_frame > 0:
            # If we're shifting the reference frame by one frame into the past, we take the last
            # state from the inputs.
            if "ref_states" in inputs:
                ref_states = inputs["ref_states"].detach()
            else:
                ref_states = torch.cat(
                    (inputs["input_states"][:, -1:], pred_states[:, :-1]),
                    axis=1,
                ).detach()
            if "ref_images" in inputs:
                ref_images = inputs["ref_images"].detach()
            else:
                ref_images = torch.cat(
                    (inputs["input_images"][:, -1:], pred_images[:, :-1]),
                    axis=1,
                ).detach()
            assert ref_images.shape == pred_images.shape
            assert ref_states.shape == pred_states.shape
        else:
            if "ref_states" in inputs:
                ref_states = inputs["ref_states"].detach()
            else:
                ref_states = pred_states.detach()

            if "ref_images" in inputs:
                ref_images = inputs["ref_images"].detach()
            else:
                ref_images = pred_images.detach()
        ref_images = ref_images[:, :, :3]

        # get masks
        proximity_mask, proximity_mask_lo = self.get_masks(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states,
            pred_actions,
            car_sizes,
            unnormalize=True,
            ref_states=ref_states,
        )
        mask_sums = proximity_mask.sum(dim=(-1, -2))
        mask_sums_lo = proximity_mask_lo.sum(dim=(-1, -2))
        # proximity_mask_lo, mask_sums_lo = proximity_mask, mask_sums

        # We impose a cost for being too far from the reference. Being too far to the side is punished more.
        diff_xy = UnitConverter.pixels_to_m(
            self.normalizer.unnormalize_states(ref_states)[..., :2]
            - self.normalizer.unnormalize_states(pred_states)[..., :2]
        ).abs()

        npred = pred_images.size(1)
        gamma_mask = (
            torch.tensor([self.config.gamma ** t for t in range(npred + 1)])
            .to(device)
            .unsqueeze(0)
        )

        proximity_cost = self.compute_proximity_cost_km(
            ref_images, proximity_mask, mask_sums
        )["costs"]

        lane_cost = self.compute_lane_cost_km(
            ref_images, proximity_mask_lo, mask_sums_lo
        )
        offroad_cost = self.compute_offroad_cost_km(
            ref_images, proximity_mask_lo, mask_sums_lo
        )

        # Multiply everything with mask_coeff used to scale up the costs that
        # depend on mask size.
        lane_loss = (
            torch.mean(lane_cost * gamma_mask[:, :npred], dim=1)
            * self.config.mask_coeff
        )
        offroad_loss = (
            torch.mean(offroad_cost * gamma_mask[:, :npred], dim=1)
            * self.config.mask_coeff
        )
        proximity_loss = (
            torch.mean(proximity_cost * gamma_mask[:, :npred], dim=1)
            * self.config.mask_coeff
        )

        reference_distance_loss = 0.0
        if self.config.lambda_r > 0:
            # Shapes are bsize, npred, 2 for each of them.
            reference_distance_cost = (
                torch.nn.functional.relu(diff_xy[:, :, 0] - 30) ** 4
                + torch.nn.functional.relu(diff_xy[:, :, 1] - 5) ** 4
            )
            reference_distance_loss = torch.mean(
                reference_distance_cost * gamma_mask[:, :npred], dim=1
            )

        # Calculating speed loss.
        # state is x, y, dx, dy, velocity
        unnormed_state = self.normalizer.unnormalize_states(pred_states)
        speed = unnormed_state[..., 2] * unnormed_state[..., 4]
        target_speed = UnitConverter.kmph_to_pixels_per_s(
            self.config.target_speed
        )
        speed_cost = (target_speed - speed).pow(2)
        # print(f"{target_speed=}, {speed[0,:]=}, {speed_cost[0]=}")
        speed_loss = torch.mean(speed_cost * gamma_mask[:, :npred], dim=1)

        traj_proximity_mask = self.get_traj_points(
            pred_images[:, :, :3].contiguous().detach(),
            pred_states,
            pred_actions,
            car_sizes,
            unnormalize=True,
            ref_states=ref_states,
        )

        # count collisions: if more then 50% of the center are covered in green, we've collided.
        collisions_sum = (traj_proximity_mask * ref_images[:, :, 1]).sum(
            dim=(2, 3)
        )
        traj_mask_sums = traj_proximity_mask.sum(dim=(2, 3))
        collisions = collisions_sum > 0.5 * traj_mask_sums

        self.overlay = proximity_mask.unsqueeze(2) * 0.85 + ref_images

        # this is to only update image every 10 times.
        if not hasattr(self, "ctr"):
            self.ctr = 0
            self.t_image = torch.zeros(10, 10)
            self.t_image_data = None

        if self.traj_landscape:
            if self.ctr % 10 == 0:
                last_image = inputs["input_images"][:, -1:, :3].repeat(
                    1, pred_states.shape[1], 1, 1, 1
                )
                last_state = inputs["input_states"].repeat(
                    1, pred_states.shape[1], 1
                )

                traj_proximity_mask = self.get_traj_points(
                    last_image,
                    pred_states,
                    pred_actions,
                    car_sizes,
                    unnormalize=True,
                    ref_states=last_state,
                )

                cost_landscape_unnormed = self.get_cost_landscape(
                    last_image[0, 0].cuda(),
                    proximity_mask[0, 0].unsqueeze(0).unsqueeze(0),
                    mask_sums[0, 0].unsqueeze(0).unsqueeze(0),
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

        return dict(
            proximity_cost=proximity_cost,
            lane_cost=lane_cost,
            offroad_cost=offroad_cost,
            lane_loss=lane_loss,
            offroad_loss=offroad_loss,
            proximity_loss=proximity_loss,
            # collisions=collisions,
            reference_distance_loss=reference_distance_loss,
            speed_loss=speed_loss,
        )

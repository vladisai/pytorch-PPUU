"""Improved costs that use edge filter and don't take max of the pixels, but
sum the values instead"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ppuu.costs.policy_costs import PolicyCost
from ppuu.data.entities import StateSequence


class PolicyCostContinuous(PolicyCost):
    @dataclass
    class Config(PolicyCost.Config):
        lambda_p: float = 4.0
        skip_contours: bool = False
        safe_factor: float = 1.5
        u_reg: float = 0.2
        lambda_a: float = 0.001

    def build_lane_offroad_mask(
        self,
        state_seq: StateSequence,
    ):
        SCALE = 0.25
        bsize, npred, nchannels, crop_h, crop_w = state_seq.images.size()

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
            .cuda()
        )
        max_y = (
            max_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        min_y = torch.ceil(
            crop_w / 2 - width
        )  # assumes other._width / 2 = self._width / 2
        min_y = (
            min_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2

        x_filter = (
            x_filter.unsqueeze(0)
            .expand(bsize * npred, crop_h)
            .type(state_seq.car_size.type())
            .cuda()
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
            .cuda()
        )
        #    y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
        y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
        y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (
            max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1)
        )
        x_filter = x_filter.cuda()
        y_filter = y_filter.cuda()
        x_filter = x_filter.type(y_filter.type())
        proximity_mask = torch.bmm(
            x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w)
        )
        proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
        proximity_mask = proximity_mask ** 2
        # mask_sum = proximity_mask.sum(dim=(-1, -2))
        # images = images.view(bsize, npred, nchannels, crop_h, crop_w)
        # costs = (
        #     torch.sum(
        #         (proximity_mask * images[:, :, 0].float()).view(
        #             bsize, npred, -1
        #         ),
        #         2,
        #     )
        #     / mask_sum
        # )
        return proximity_mask
        # return costs.view(bsize, npred), proximity_mask

    def build_car_proximity_mask(
        self,
        state_seq: StateSequence,
        unnormalize: bool = False,
    ) -> torch.Tensor:
        device = state_seq.images.device

        SCALE = 0.25
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
            torch.abs(speed) * self.config.safe_factor + (1 * 24 / 3.7) * SCALE
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
            .type(state_seq.car_size.type())
            .to(device)
        )
        max_y = (
            max_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .type(state_seq.car_size.type())
            .to(device)
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
            .to(device)
        )
        torch.set_default_tensor_type(torch.FloatTensor)

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
        x_filter = torch.max(x_filter, min_x.view(bsize * npred, 1))

        x_filter = (x_filter - min_x.view(bsize * npred, 1)) / (
            max_x - min_x
        ).view(bsize * npred, 1)
        y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
        y_filter = (
            y_filter.view(1, crop_w)
            .expand(bsize * npred, crop_w)
            .type(state_seq.car_size.type())
            .to(device)
        )
        y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
        y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
        y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (
            max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1)
        )
        x_filter = x_filter.type(state_seq.car_size.type()).to(device)
        y_filter = y_filter.type(state_seq.car_size.type()).to(device)
        proximity_mask = torch.bmm(
            x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w)
        )
        proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
        images = images.view(-1, nchannels, crop_h, crop_w)
        # pre_max = (proximity_mask * (green_image ** 2))
        # green_contours[green_contours < 0.5] = 0
        proximity_mask = proximity_mask ** 2
        return proximity_mask
        # mask_sum = proximity_mask.sum(dim=(-1, -2))
        # pre_max = proximity_mask * (green_contours ** 2)
        # # costs = torch.max(pre_max.view(bsize, npred, -1), 2)[0]
        # costs = torch.sum(pre_max.view(bsize, npred, -1), 2) / mask_sum

        # images = images.view(bsize, npred, nchannels, crop_h, crop_w)
        # green_image = images[:, :, 1].float()
        # pre_max_old = proximity_mask * green_image
        # costs_old = torch.max(pre_max_old.view(bsize, npred, -1), 2)[0]

        # result = {}
        # result["costs"] = costs
        # result["costs_old"] = costs_old
        # result["masks"] = proximity_mask
        # result["pre_max"] = pre_max
        # result["contours"] = green_contours

        # return result

    def compute_contours(self, images: torch.Tensor) -> torch.Tensor:
        """Computes contours of the green channel.
        The idea is to get only edges of the cars so that later
        when we do summing the size of the cars doesn't affect our behavior.
        Input and output are both of shape bsize, npred, height, width
        """
        device = images.device
        horizontal_filter = torch.tensor(
            [[[-1.0], [1.0]]],
            device=device,
        )
        horizontal_filter = horizontal_filter.expand(1, 1, 2, 1)
        vertical_filter = torch.tensor([[1.0, -1.0]], device=device).view(
            1, 1, 2
        )
        vertical_filter = vertical_filter.expand(1, 1, 1, 2)

        original_images_shape = images.shape
        # new shape is -1, 1, height, width
        images = images.view(-1, *images.shape[-2:]).unsqueeze(-3)

        horizontal = F.conv2d(
            images, horizontal_filter, stride=1, padding=(1, 0)
        )
        horizontal = horizontal[:, :, :-1, :]

        vertical = F.conv2d(images, vertical_filter, stride=1, padding=(0, 1))
        vertical = vertical[:, :, :, :-1]

        _, _, height, width = horizontal.shape

        horizontal_mask = torch.ones((1, 1, height, width), device=device)
        horizontal_mask[:, :, : (height // 2), :] = -1
        horizontal_masked = F.relu(horizontal_mask * horizontal)

        vertical_mask = torch.ones((1, 1, height, width), device=device)
        vertical_mask[:, :, :, (width // 2) :] = -1
        vertical_masked = F.relu(vertical_mask * vertical)

        result = vertical_masked[:][:] + horizontal_masked[:][:]
        result = result.view(*original_images_shape)
        return result

    def _multiply_masks(
        self, images: torch.Tensor, proximity_mask: torch.Tensor
    ) -> torch.Tensor:
        """Given an image of shape bsize, npred, height, width and a mask
        of same shape, multiplies them, and sums them up. This is the main
        difference from the vanilla version of the cost.
        Returns tensor of shape bsize, npred.
        """
        mask_sum = proximity_mask.sum(dim=(-1, -2))
        return (
            torch.sum(
                (proximity_mask * images.float()).view(*images.shape[:2], -1),
                2,
            ).view(*images.shape[:2])
            / mask_sum
        )

    def compute_proximity_cost(
        self,
        state_seq: StateSequence,
        proximity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the proximity cost using the provided mask
        Returns a tensor of shape (batch_size, npred).
        """
        images = state_seq.images[:, :, 1]
        if not self.config.skip_contours:
            images = self.compute_contours(images)
        return self._multiply_masks(images, proximity_mask)

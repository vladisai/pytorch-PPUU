from __future__ import annotations

from typing import NamedTuple

import torch


class StateSequence(NamedTuple):
    """A class to hold info about state sequences used throughout this codebase."""

    images: torch.Tensor  # of shape (..., npred, 3 or 4, 117, 24)
    states: torch.Tensor  # of shape (..., npred, 5)
    car_size: torch.Tensor  # of shape (..., 2)
    ego_car_image: torch.Tensor  # of shape (..., 3, 117, 24)

    def with_ego(self) -> StateSequence:
        """This creates another sequence with modified image to contain the images
        of the ego car.
        """
        assert (
            self.images.shape[-3] == 3
        ), "images channels is already greater than 3"
        ego_car_new_shape = [*self.images.shape]
        ego_car_new_shape[-3] = 1

        if len(self.images.shape) == 5:
            assert (
                self.images.shape[0] == self.ego_car_image.shape[0]
            ), "the batch sizes dont't match"

        ego_car = self.ego_car_image[..., 2, :, :]
        ego_car = ego_car.unsqueeze(-3)  # add channel back
        ego_car = ego_car.unsqueeze(-4).expand(ego_car_new_shape)  # add  npred

        images_with_ego = torch.cat(
            (self.images.clone(), ego_car), dim=-3
        ).contiguous()

        return StateSequence(
            images_with_ego, self.states, self.car_size, self.ego_car_image
        )

    def without_ego(self) -> StateSequence:
        return StateSequence(
            self.images[..., :3, :, :].contiguous(),
            self.states,
            self.car_size,
            self.ego_car_image,
        )

    def shift_add(
        self, image: torch.Tensor, state: torch.Tensor
    ) -> StateSequence:
        """Builds another sequence with image and state added to the sequence to
        the end, and the first element removed.
        This is useful when doing unfolding with the forward model.
        """
        if self.images.shape[-3] == 4 and image.shape[-3] == 3:
            image = torch.cat(
                (
                    image.clone(),
                    self.ego_car_image[..., 2, :, :]
                    .unsqueeze(-3)
                    .unsqueeze(-3),
                ),
                dim=-3,
            )

        # ... allows us to be agnostic to presence of batches.
        if len(image.shape) < len(self.images.shape):
            image = image.unsqueeze(-4)

        new_images = torch.cat((self.images[..., 1:, :, :, :], image), dim=-4)
        new_states = torch.cat(
            (self.states[..., 1:, :], state.unsqueeze(-2)), dim=-2
        )
        return StateSequence(
            new_images, new_states, self.car_size, self.ego_car_image
        )

    def to(self, device: torch.device) -> StateSequence:
        """Creates a new StateSequence with all fields moved to specified device"""
        return StateSequence(
            self.images.to(device),
            self.states.to(device),
            self.car_size.to(device),
            self.ego_car_image.to(device),
        )

    def cuda(self) -> StateSequence:
        """Creates a new DatasetSample with all fields moved to cuda"""
        return self.to(torch.device("cuda"))

    def cpu(self) -> StateSequence:
        """Creates a new DatasetSample with all fields moved to cuda"""
        return self.to(torch.device("cpu"))

    def map(self, f) -> StateSequence:
        return StateSequence(*map(f, self))


class DatasetSample(NamedTuple):
    """Dataset sample contains one information about one example from the training
    set, i.e. a snippet of an episode split into conditional states and target
    states, and also some information about the split itself.
    """

    conditional_state_seq: StateSequence
    target_state_seq: StateSequence
    target_action_seq: torch.Tensor
    sample_split_id: torch.Tensor
    episode_id: int
    timestep: int

    def cuda(self) -> DatasetSample:
        """Creates a new DatasetSample with all fields moved to cuda"""
        return self.to(torch.device("cuda"))

    def cpu(self) -> DatasetSample:
        """Creates a new DatasetSample with all fields moved to cuda"""
        return self.to(torch.device("cpu"))

    def to(self, device: torch.device) -> DatasetSample:
        """Creates a new DatasetSample with all fields moved to specified device"""
        return DatasetSample(
            self.conditional_state_seq.to(device),
            self.target_state_seq.to(device),
            self.target_action_seq.to(device),
            self.sample_split_id,
            self.episode_id,
            self.timestep,
        )

    @classmethod
    def from_tuple(cls, t) -> DatasetSample:
        # NEXT finish from tuple method and put into lightning.
        return DatasetSample(
            StateSequence(*t[0]),
            StateSequence(*t[1]),
            *t[2:],
        )

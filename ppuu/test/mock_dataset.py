import torch

from ppuu.data.entities import DatasetSample, StateSequence


class MockDataset:
    def __init__(self, length=100, ncond=20, npred=30):
        self.length = length
        self.ncond = ncond
        self.npred = npred

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        cnd_images = torch.randn(self.ncond, 3, 117, 24)
        target_images = torch.randn(self.npred, 3, 117, 24)

        cnd_states = torch.randn(self.ncond, 5)
        target_states = torch.randn(self.npred, 5)

        actions = torch.randn(self.npred, 2)
        ego_car = torch.randn(3, 117, 24)

        car_size = torch.randn(2)

        conditional_state_seq = StateSequence(
            cnd_images, cnd_states, car_size, ego_car
        )

        target_state_seq = StateSequence(
            target_images, target_states, car_size, ego_car
        )

        return DatasetSample(
            conditional_state_seq,
            target_state_seq,
            actions,
            sample_split_id=torch.tensor(0),
            episode_id=0,
            timestep=0,
        )


def get_mock_dataloader(length=100, batch_size=2):
    return torch.utils.data.DataLoader(MockDataset(length), batch_size)

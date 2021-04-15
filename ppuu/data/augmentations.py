import numpy as np
import torch
import torch.nn.functional as F


class Augmenter:
    def __init__(self, std=0.07, p=0.0):
        self.std = std
        self.p = p

    def __call__(self, batch):
        if self.p == 0:
            return batch
        new_batch = []
        for i in range(batch.shape[0]):
            new_sample = batch[i]
            if torch.rand(1).item() < self.p:
                size = np.random.randint(1, 3)
                augmentations = np.random.choice(
                    ["blur", "noise"], replace=False, size=size
                )
                for aug in augmentations:
                    if aug == "blur":
                        new_sample = self.random_blur(new_sample)
                    elif aug == "noise":
                        new_sample = self.random_noise(new_sample)
            new_batch.append(new_sample)
        return torch.stack(new_batch)

    def random_blur(self, batch):
        f = self.get_gaussian_filter(np.random.rand() * 0.8 + 0.1, 3)
        f = f.to(device=batch.device)
        return F.conv2d(
            batch,
            f,
            groups=4,
            stride=1,
            padding=1,
        )

    def random_noise(self, batch):
        return batch + torch.normal(
            0, np.random.rand() * self.std, batch.shape, device=batch.device
        )

    @staticmethod
    def get_gaussian_filter(std=1, kernel_size=3):
        values = np.zeros((kernel_size, kernel_size))
        margin = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - margin
                y = j - margin
                values[i][j] = (
                    1
                    / (2 * np.pi * std ** 2)
                    * np.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
                ).item()
        values = values / values.sum()
        return torch.from_numpy(values).expand(4, 1, 3, 3).float()

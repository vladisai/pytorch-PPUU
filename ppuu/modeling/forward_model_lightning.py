from typing import Union

import torch

from ppuu.lightning_modules.fm import FM


class ForwardModelV2(torch.nn.Module):
    def __init__(self, file_path):
        super().__init__()
        self.module = FM.load_from_checkpoint(file_path)

    def __getattr__(self, name):
        """Delegate everything to forward_model"""
        return getattr(self._modules['module'].model, name)

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader

from ppuu.data.dataloader import DataStore, Dataset


def _worker_init_fn(index):
    info = torch.utils.data.get_worker_info()
    info.dataset.random.seed(info.seed)


class NGSIMDataModule(pl.LightningDataModule):
    """
    Class that implements datamodule interface for lightning
    for ngsim dataset.

    NOTE:
        Using more than 0 workers seems to break when using multiple nodes,
        as shared memory begins to leak and everything breaks.
    """

    def __init__(
        self,
        path,
        epoch_size,
        validation_size,
        batch_size,
        shift=False,
        random_actions=False,
        ncond=20,
        npred=30,
        val_npred=30,
        workers=0,
        diffs=False,
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.validation_size = validation_size
        self.workers = workers
        self.shift = shift
        self.random_actions = random_actions
        self.npred = npred
        self.ncond = ncond
        self.val_npred = val_npred
        self.diffs = diffs

    def setup(self, stage=None):
        self.data_store = DataStore(self.path)
        samples_in_epoch = self.epoch_size * self.batch_size
        samples_in_validation = self.validation_size * self.batch_size
        self.train_dataset = Dataset(
            self.data_store,
            "train",
            self.ncond,
            self.npred,
            size=samples_in_epoch,
            shift=self.shift,
            random_actions=self.random_actions,
            state_diffs=self.diffs,
        )
        self.val_dataset = Dataset(
            self.data_store,
            "val",
            self.ncond,
            self.val_npred,
            size=samples_in_validation,
            shift=self.shift,
            random_actions=self.random_actions,
            state_diffs=self.diffs,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            worker_init_fn=_worker_init_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            worker_init_fn=_worker_init_fn,
        )
        return loader

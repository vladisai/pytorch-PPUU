import dataclasses
from collections import deque
from dataclasses import dataclass
from typing import Union

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from ppuu import configs
from ppuu.data.dataloader import Normalizer
from ppuu.data.entities import DatasetSample
from ppuu.modeling import get_fm
from ppuu.modeling.km import StatePredictor


@dataclass
class ModelConfig(configs.ConfigBase):
    n_hidden: int = 256
    n_feature: int = 256

    n_inputs: int = 4
    n_actions: int = 2
    height: int = 117
    width: int = 24
    h_height: int = 14
    h_width: int = 3
    hidden_size: int = n_feature * h_height * h_width
    layers: int = 3

    dropout: float = 0.1
    z_dropout: float = 0.5
    nfeature: int = 256
    nz: int = 32

    ncond: int = 20

    model_type: str = "fm"
    fm_type: str = "km"

    beta: float = 1e-06

    huber_loss: bool = False

    predict_state: bool = False

    checkpoint: Union[str, None] = None


@dataclass
class TrainingConfig(configs.TrainingConfig):
    decay: float = 1
    decay_period: int = 1

    n_cond: int = 20
    n_pred: int = 20

    auto_enable_latent: bool = False

    enable_latent: bool = False
    batch_size: int = 64

    epoch_size: int = 500
    validation_steps: int = 10000
    validation_period: int = 1
    n_epochs: int = 1600
    rebalance: bool = False

    def auto_batch_size(self):
        if self.batch_size == -1:
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.batch_size = int((gpu_gb / 4) * 64)
            print("auto batch size is set to", self.batch_size)
            self.validation_size = int(self.validation_steps / self.batch_size)
            print("auto validdation size is set to", self.validation_size)
        self.auto_n_epochs()


class PlateauDetector(object):
    def __init__(self, threshold: float = 1e-4, patience: int = 10):
        self.threshold = threshold
        self.patience = patience
        self.last_values = deque(maxlen=patience)

    def update(self, value):
        self.last_values.append(value)

    def detected(self):
        if len(self.last_values) < self.patience:
            return False
        diff = self.last_values[0] - self.last_values[-1]
        return diff < self.threshold


class FM(pl.LightningModule):
    @dataclass
    class Config(configs.ConfigBase):
        model: ModelConfig = ModelConfig()
        training: TrainingConfig = TrainingConfig()

    def __init__(
        self,
        hparams=None,
    ):
        super().__init__()
        self.set_hparams(hparams)
        self.state_predictor = StatePredictor(self.config.training.diffs, None)
        self.model = get_fm(self.config.model.fm_type)(
            layers=self.config.model.layers,
            nfeature=self.config.model.nfeature,
            dropout=self.config.model.dropout,
            h_height=self.config.model.h_height,
            h_width=self.config.model.h_width,
            height=self.config.model.height,
            width=self.config.model.width,
            n_actions=self.config.model.n_actions,
            hidden_size=self.config.model.hidden_size,
            ncond=self.config.model.ncond,
            nz=self.config.model.nz,
            enable_kld=True,
            enable_latent=self.config.training.enable_latent,
            state_predictor=self.state_predictor,
        )
        self.plateau_detector = PlateauDetector(1e-3, 5)

        if self.config.model.checkpoint is not None:
            checkpoint = torch.load(self.config.model.checkpoint)
            self.load_state_dict(checkpoint["state_dict"])

    def forward(self, batch):
        predictions = self.model(
            input_state_seq=batch.conditional_state_seq,
            actions=batch.target_action_seq,
            target_state_seq=batch.target_state_seq,
            z_dropout=self.config.model.z_dropout,
        )
        return predictions

    def on_train_start(self):
        """
        The setup below happens after everything was moved
        to the correct device, and after the data was loaded.
        """
        self._setup_normalizer(self.trainer.datamodule.data_store.stats)

    def _setup_normalizer(self, stats):
        self.normalizer = Normalizer(stats)
        self.state_predictor.normalizer = self.normalizer

    def shared_step(self, batch: DatasetSample):
        predictions = self(batch)
        if self.config.model.huber_loss:
            states_loss = F.smooth_l1_loss(
                batch.target_state_seq.states, predictions.state_seq.states
            )
        else:
            states_loss = F.mse_loss(
                batch.target_state_seq.states, predictions.state_seq.states
            )
        images_loss = (
            (
                (batch.target_state_seq.images - predictions.state_seq.images)
                ** 2
            )
            .view(*batch.target_action_seq.shape[:2], -1)
            .mean(dim=-1)
        )
        if self.config.training.rebalance:
            ranking = torch.clamp(
                torch.exp(0.5 * batch.target_action_seq.abs()[:, :, 1].pow(2)),
                min=1,
                max=10,
            )
            rebalanced_images_loss = (images_loss * ranking).mean()
        else:
            rebalanced_images_loss = -1

        images_loss = images_loss.mean()

        p_loss = predictions.p_loss
        return states_loss, images_loss, rebalanced_images_loss, p_loss

    @property
    def sample_step(self):
        return (
            self.trainer.global_step
            * self.config.training.batch_size
            * self.config.training.num_nodes
            * self.config.training.gpus
        )

    def training_step(self, batch, batch_idx):
        batch = DatasetSample.from_tuple(batch)
        (
            states_loss,
            images_loss,
            rebalanced_images_loss,
            p_loss,
        ) = self.shared_step(batch)
        if self.config.training.rebalance:
            loss = (
                rebalanced_images_loss
                + states_loss
                + self.config.model.beta * p_loss
            )
        else:
            loss = images_loss + states_loss + self.config.model.beta * p_loss

        if torch.isnan(loss).any():
            loss = torch.tensor(0.0, requires_grad=True).to(self.device) * 5
            print("NaN loss!")

        logs = {
            "states_loss": states_loss,
            "images_loss": images_loss,
            "p_loss": p_loss,
            "total_loss": loss,
        }
        if self.config.training.rebalance:
            logs["rebalanced_images_loss"] = rebalanced_images_loss

        res = logs["total_loss"]
        for k in logs:
            self.log(k, logs[k], on_step=True, prog_bar=True, logger=True)
        return res

    def validation_step(self, batch, batch_idx):
        batch = DatasetSample.from_tuple(batch)
        shared_losses = self.shared_step(batch)

        predictions = self.model.unfold(
            batch.conditional_state_seq, batch.target_action_seq
        )
        states_loss = F.mse_loss(
            batch.target_state_seq.states, predictions.state_seq.states
        )
        images_loss = F.mse_loss(
            batch.target_state_seq.images, predictions.state_seq.images
        )
        loss = images_loss + states_loss
        logs = {
            "val_unfold_states_loss": states_loss,
            "val_unfold_images_loss": images_loss,
            "val_unfold_total_loss": loss,
            "val_states_loss": shared_losses[0],
            "val_images_loss": shared_losses[1],
            "val_rebalanced_images_loss": shared_losses[2],
            "val_p_loss": shared_losses[3],
        }
        res = loss
        for k in logs:
            self.log(k, logs[k], on_epoch=True, logger=True)

        return res

    def validation_epoch_end(self, res):
        avg_loss = torch.stack(res).mean()
        self.log("sample_step", self.sample_step)
        self._check_plateau(avg_loss)
        return avg_loss

    @pl.loggers.base.rank_zero_only
    def _check_plateau(self, value):
        self.plateau_detector.update(value)
        if (
            self.config.training.auto_enable_latent
            and self.plateau_detector.detected()
        ):
            self.model.set_enable_latent(True)
            print("enabled the latent!")
        self.logger.experiment.log(
            {
                "latent_enabled": float(self.model.enable_latent)
                + torch.rand(1).item() * 0.05
            }
        )

    def set_hparams(self, hparams=None):
        if hparams is None:
            hparams = FM.Config()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
            self.config = FM.Config.parse_from_dict(hparams)
        else:
            self.hparams.update(dataclasses.asdict(hparams))
            self.config = hparams

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.config.training.learning_rate
            * self.config.training.gpus
            * self.config.training.num_nodes
            * (64 / self.config.training.batch_size),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            self.config.training.decay_period,
            self.config.training.decay,
        )
        return [optimizer], [scheduler]

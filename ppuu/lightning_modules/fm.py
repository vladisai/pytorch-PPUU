import dataclasses
from dataclasses import dataclass
from collections import deque

import pytorch_lightning as pl

import torch
from torch.nn import functional as F

from ppuu.modeling import FwdCNN_VAE
from ppuu import configs


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
    nfeature: float = 256
    nz: float = 32

    ncond: int = 20

    model_type: str = "fm"

    beta: float = 1e-06

    huber_loss: bool = False

@dataclass
class TrainingConfig(configs.TrainingConfig):
    decay: float = 1
    decay_period: int = 1

    n_cond: int = 20
    n_pred: int = 20

    auto_enable_latent: bool = False


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
        self, hparams=None,
    ):
        super().__init__()
        self.set_hparams(hparams)
        self.model = FwdCNN_VAE(
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
            enable_latent=False,
        )
        self.plateau_detector = PlateauDetector(1e-3, 5)

    def forward(self, batch):
        predictions = self.model(
            inputs=(batch["input_images"], batch["input_states"]),
            actions=batch["actions"],
            targets=(batch["target_images"], batch["target_states"]),
            z_dropout=self.config.model.z_dropout,
        )
        return predictions

    def shared_step(self, batch):
        predictions = self(batch)
        if self.config.model.huber_loss:
            states_loss = F.smooth_l1_loss(
                batch["target_states"], predictions.pred_states
            )
        else:
            states_loss = F.mse_loss(
                batch["target_states"], predictions.pred_states
            )
        images_loss = F.mse_loss(
            batch["target_images"], predictions.pred_images
        )
        p_loss = predictions.p_loss
        return states_loss, images_loss, p_loss

    def training_step(self, batch, batch_idx):
        states_loss, images_loss, p_loss = self.shared_step(batch)
        loss = images_loss + states_loss + self.config.model.beta * p_loss

        if torch.isnan(loss).any():
            loss = torch.tensor(0.0, requires_grad=True) * 5
            print('NaN loss!')

        logs = {
            "states_loss": states_loss,
            "images_loss": images_loss,
            "p_loss": p_loss,
            "total_loss": loss,
        }
        res = pl.TrainResult(loss)
        for k in logs:
            res.log(
                k,
                logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return res

    def validation_step(self, batch, batch_idx):
        predictions = self.model.unfold(batch["actions"], batch)
        states_loss = F.mse_loss(
            batch["target_states"], predictions["pred_states"]
        )
        images_loss = F.mse_loss(
            batch["target_images"], predictions["pred_images"]
        )
        loss = images_loss + states_loss
        logs = {
            "val_states_loss": states_loss,
            "val_images_loss": images_loss,
            "val_total_loss": loss,
        }
        res = pl.EvalResult(loss)
        res.log_dict(
            logs, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        return res

    def validation_epoch_end(self, res):
        res.dp_reduce()
        self._check_plateau(res["early_stop_on"].detach())
        return res

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
            self.hparams = hparams
            self.config = FM.Config.parse_from_dict(hparams)
        else:
            self.hparams = dataclasses.asdict(hparams)
            self.config = hparams

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.config.training.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            self.config.training.decay_period,
            self.config.training.decay,
        )
        return [optimizer], [scheduler]

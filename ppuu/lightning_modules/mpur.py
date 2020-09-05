"""Train a policy / controller"""
import dataclasses
from dataclasses import dataclass
import hashlib


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ppuu.data import DataStore, Dataset, EvaluationDataset
from ppuu.costs import PolicyCost, PolicyCostContinuous
from ppuu import configs
from ppuu.modeling import policy_models
from ppuu.modeling.forward_models import ForwardModel
from ppuu.eval import PolicyEvaluator
from ppuu.data import Augmenter
from ppuu.modeling.mixout import MixoutWrapper


def inject(cost_type=PolicyCost, fm_type=ForwardModel):
    """ This injector allows to customize lightning modules with custom cost
    class and forward model class (or more, if extended).  It injects these
    types and creates a config dataclass that contains configs for all the
    components of this lightning module. Any new module has to be injected
    into, as without it the class doesn't know which cost or forward model to
    use, and also has no config.

    The config class has to be added to the global scope through globals().
    It's a hack to make it pickleable for multiprocessing later.
    If the MPURModule has to be pickleable too, we'd need to put it into
    global scope too.
    """

    def wrapper(cls_):
        h = hashlib.md5(
            (cost_type.__qualname__ + fm_type.__qualname__).encode()
        ).hexdigest()[:7]
        suffix = f"{cls_.__name__}_{cost_type.__name__}_{fm_type.__name__}_{h}"
        config_name = f"config_{suffix}"

        class Cls(cls_):
            CostType = cost_type
            ForwardModelType = fm_type

            @dataclass
            class Config(configs.ConfigBase):
                model_config: cls_.ModelConfig = cls_.ModelConfig()
                cost_config: cost_type.Config = cost_type.Config()
                training_config: cls_.TrainingConfig = cls_.TrainingConfig()

        Cls.Config.__qualname__ = config_name
        Cls.Config.__name__ = config_name
        globals()[config_name] = Cls.Config
        return Cls

    return wrapper


@inject(cost_type=PolicyCost, fm_type=ForwardModel)
class MPURModule(pl.LightningModule):
    @dataclass
    class ModelConfig(configs.ModelConfig):
        forward_model_path: str = "/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/offroad/model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step400000.model"  # noqa: E501

        n_cond: int = 20
        n_pred: int = 30
        n_hidden: int = 256
        n_feature: int = 256

        n_inputs: int = 4
        n_actions: int = 2
        height: int = 117
        width: int = 24
        h_height: int = 14
        h_width: int = 3
        hidden_size: int = n_feature * h_height * h_width

    TrainingConfig = configs.TrainingConfig

    def __init__(self, hparams=None):
        super().__init__()
        self.set_hparams(hparams)

        self.forward_model = self.ForwardModelType(
            self.config.model_config.forward_model_path
        )
        self.policy_model = policy_models.DeterministicPolicy(
            n_cond=self.config.model_config.n_cond,
            n_feature=self.config.model_config.n_feature,
            n_actions=self.config.model_config.n_actions,
            h_height=self.config.model_config.h_height,
            h_width=self.config.model_config.h_width,
            n_hidden=self.config.model_config.n_hidden,
        )
        self.policy_model.train()

        if self.config.model_config.checkpoint is not None:
            checkpoint = torch.load(self.config.model_config.checkpoint)
            self.load_state_dict(checkpoint["state_dict"])

        if self.config.training_config.freeze_encoder:
            for name, p in self.policy_model.named_parameters():
                if "fc" not in name:
                    p.requires_grad = False

        self.augmenter = Augmenter(
            self.config.training_config.noise_augmentation_std,
            self.config.training_config.noise_augmentation_p,
        )

    def set_hparams(self, hparams=None):
        if hparams is None:
            hparams = MPURModule.Config()
        if isinstance(hparams, dict):
            self.hparams = hparams
            self.config = MPURModule.Config.parse_from_dict(hparams)
        else:
            self.hparams = dataclasses.asdict(hparams)
            self.config = hparams

    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold(
            self.policy_model, batch, augmenter=self.augmenter
        )
        return predictions

    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = self.policy_cost.calculate_cost(batch, predictions)
        logs = loss.copy()
        logs["action_norm"] = (
            predictions["pred_actions"].norm(2, 2).pow(2).mean()
        )
        res = pl.TrainResult(loss["policy_loss"])
        res.log_dict(
            logs, on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )
        return res

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = self.policy_cost.calculate_cost(batch, predictions)
        res = pl.EvalResult(loss["policy_loss"])
        res.log_dict(
            loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )
        return res

    def validation_epoch_end(self, res):
        res.dp_reduce()
        if self.config.training_config.validation_eval:
            eval_results = self.evaluator.evaluate(self)
            # eval_results = {'stats': {'success_rate': 0.0}}
            res.log_dict(
                eval_results["stats"],
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return res

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.policy_model.parameters(),
            self.config.training_config.learning_rate,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.90)
        return [optimizer], [scheduler]

    def on_train_start(self):
        """
        The setup below happens after everything was moved
        to the correct device, and after the data was loaded.
        """
        self._setup_forward_model()
        self._setup_mixout()
        self._setup_policy_cost()
        self._setup_episode_evaluator()

    def _setup_forward_model(self):
        self.forward_model.eval()
        self.forward_model.device = self.device
        self.forward_model.to(self.device)

    def _setup_policy_cost(self):
        self.policy_cost = self.CostType(
            self.config.cost_config,
            self.forward_model,
            self.trainer.datamodule.data_store.stats,
        )
        self.policy_cost.estimate_uncertainty_stats(self.train_dataloader())

    def _setup_mixout(self):
        if self.config.training_config.mixout_p is not None:
            # self.policy_model = MixoutDeterministicPolicy(
            #     self.policy_model, p=self.config.training_config.mixout_p
            # )
            self.policy_model = self.policy_model.apply(
                lambda x: MixoutWrapper(
                    x, self.config.training_config.mixout_p
                )
            )

    def _setup_episode_evaluator(self):
        self.eval_dataset = EvaluationDataset.from_data_store(
            self.trainer.datamodule.data_store, split="val", size_cap=25
        )
        self.evaluator = PolicyEvaluator(
            self.eval_dataset,
            num_processes=5,
            build_gradients=False,
            return_episode_data=False,
            enable_logging=False,
        )

    @classmethod
    def _load_model_state(cls, checkpoint, *args, **kwargs):
        copy = dict()
        for k in checkpoint["state_dict"]:
            # Adjust for nested forward model.
            if k.startswith("forward_model") and not k.startswith(
                "forward_model.forward_model."
            ):
                copy["forward_model." + k] = checkpoint["state_dict"][k]
            elif k.startswith("policy_model.original_model"):
                if "fc" not in k:
                    copy[k.replace(".original_model", "")] = checkpoint[
                        "state_dict"
                    ][k]
            elif "target" not in k:
                copy[k] = checkpoint["state_dict"][k]

            # Change for mixout network.
        checkpoint["state_dict"] = copy
        return super()._load_model_state(checkpoint, *args, **kwargs)


@inject(cost_type=PolicyCostContinuous, fm_type=ForwardModel)
class MPURContinuousModule(MPURModule):
    pass

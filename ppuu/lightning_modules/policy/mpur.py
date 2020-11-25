"""Train a policy / controller"""
import dataclasses
from dataclasses import dataclass
import hashlib


import torch
import torch.optim as optim
import pytorch_lightning as pl

from ppuu.data import EvaluationDataset
from ppuu.costs import PolicyCost, PolicyCostContinuous
from ppuu import configs
from ppuu.modeling import policy_models
from ppuu.wrappers import ForwardModel
from ppuu.eval import PolicyEvaluator
from ppuu.data import Augmenter
from ppuu.modeling.mixout import MixoutWrapper
from ppuu.lightning_modules.fm import FM


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

        turn_power: int = 1

    TrainingConfig = configs.TrainingConfig

    def __init__(self, hparams=None):
        super().__init__()
        self.set_hparams(hparams)

        self.forward_model = self.ForwardModelType(
            self.config.model_config.forward_model_path, self.config.training_config.diffs
        )

        self.policy_model = policy_models.DeterministicPolicy(
            n_cond=self.config.model_config.n_cond,
            n_feature=self.config.model_config.n_feature,
            n_actions=self.config.model_config.n_actions,
            h_height=self.config.model_config.h_height,
            h_width=self.config.model_config.h_width,
            n_hidden=self.config.model_config.n_hidden,
            diffs=self.config.training_config.diffs,
            turn_power=self.config.model_config.turn_power,
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
        self.nan_ctr = 0

    def set_hparams(self, hparams=None):
        if hparams is None:
            hparams = self.Config()
        if isinstance(hparams, dict):
            self.hparams = hparams
            self.config = self.Config.parse_from_dict(hparams)
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
        res = loss["policy_loss"]
        for k in logs:
            self.log(
                "train_" + k,
                logs[k],
                on_step=True,
                logger=True,
                prog_bar=True,
            )
        return res

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = self.policy_cost.calculate_cost(batch, predictions)
        res = loss["policy_loss"]
        for k in loss:
            self.log(
                "val_" + k, loss[k], on_epoch=True, logger=True,
            )
        return res

    @property
    def sample_step(self):
        return (
            self.trainer.global_step
            * self.config.training_config.batch_size
            * self.config.training_config.num_nodes
            * self.config.training_config.gpus
        )

    def validation_epoch_end(self, res):
        self.log("sample_step", self.sample_step)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.policy_model.parameters(),
            self.config.training_config.learning_rate
            * self.config.training_config.gpus
            * self.config.training_config.num_nodes
            * (6 / self.config.training_config.batch_size),
        )
        if self.config.training_config.scheduler:
            # we want to have 0.1 learning rate after 70% of training
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(self.config.training_config.n_epochs * 0.7),
                gamma=0.1,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

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
            self.mixout_wrapper = MixoutWrapper(
                self.config.training_config.mixout_p
            )
            self.policy_model = self.policy_model.apply(
                lambda x: self.mixout_wrapper(x)
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

    # @classmethod
    # def _load_model_state(cls, checkpoint, *args, **kwargs):
    #     copy = dict()
    #     for k in checkpoint["state_dict"]:
    #         # Adjust for nested forward model.
    #         if k.startswith("forward_model") and not k.startswith(
    #             "forward_model.forward_model."
    #         ):
    #             copy["forward_model." + k] = checkpoint["state_dict"][k]
    #         elif k.startswith("policy_model.original_model"):
    #             if "fc" not in k:
    #                 copy[k.replace(".original_model", "")] = checkpoint[
    #                     "state_dict"
    #                 ][k]
    #         elif "target" not in k:
    #             copy[k] = checkpoint["state_dict"][k]

    #         # Change for mixout network.
    #     checkpoint["state_dict"] = copy
    #     return super()._load_model_state(checkpoint, *args, **kwargs)


@inject(cost_type=PolicyCostContinuous, fm_type=ForwardModel)
class MPURContinuousModule(MPURModule):
    pass


class ForwardModelV2(torch.nn.Module):
    def __init__(self, file_path):
        super().__init__()
        module = FM.load_from_checkpoint(file_path)
        module.model.enable_latent = True
        self.module = module

    def __getattr__(self, name):
        """Delegate everything to forward_model"""
        return getattr(self._modules["module"].model, name)


class ForwardModelV3(torch.nn.Module):
    """FM with no action and diff"""

    def __init__(self, file_path, diffs):
        super().__init__()
        m_config = FM.Config()
        m_config.model.fm_type = "km_no_action"
        m_config.model.checkpoint = file_path
        m_config.training.enable_latent = True
        m_config.training.diffs = diffs
        module = FM(m_config)
        self.module = module

    def __getattr__(self, name):
        """Delegate everything to forward_model"""
        return getattr(self._modules["module"].model, name)


@inject(cost_type=PolicyCostContinuous, fm_type=ForwardModelV2)
class MPURContinuousV2Module(MPURContinuousModule):
    @dataclass
    class ModelConfig(MPURContinuousModule.ModelConfig):
        forward_model_path: str = "/home/us441/nvidia-collab/vlad/results/fm/fm_km_5_states_resume_lower_lr/seed=42/checkpoints/epoch=23_success_rate=0.ckpt"

    # @classmethod
    # def _load_model_state(cls, checkpoint, *args, **kwargs):
    #     return super(MPURModule, cls)._load_model_state(checkpoint, *args, **kwargs)


# noqa: E501


@inject(cost_type=PolicyCost, fm_type=ForwardModelV2)
class MPURVanillaV2Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        forward_model_path: str = "/home/us441/nvidia-collab/vlad/results/fm/fm_km_5_states_resume_lower_lr/seed=42/checkpoints/epoch=23_success_rate=0.ckpt"


@inject(cost_type=PolicyCost, fm_type=ForwardModelV3)
class MPURVanillaV3Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        forward_model_path: str = "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/fm_km_no_action_diff_64_even_lower_lr/seed=42/checkpoints/last.ckpt"


@inject(cost_type=PolicyCostContinuous, fm_type=ForwardModelV3)
class MPURContinuousV3Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        forward_model_path: str = "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/fm_km_no_action_diff_64_even_lower_lr/seed=42/checkpoints/last.ckpt"

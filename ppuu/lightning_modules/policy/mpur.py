"""Train a policy / controller"""
import dataclasses
from dataclasses import dataclass
import hashlib

import wandb

import torch
import torch.optim as optim
import pytorch_lightning as pl

from omegaconf import MISSING

from ppuu.data import EvaluationDataset
from ppuu.costs import PolicyCost, PolicyCostContinuous
from ppuu import configs
from ppuu.modeling import policy_models
from ppuu.wrappers import ForwardModel
from ppuu.eval import PolicyEvaluator
from ppuu.data import Augmenter
from ppuu.modeling.mixout import MixoutWrapper
from ppuu.lightning_modules.fm import FM

from ppuu.data.dataloader import Normalizer


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
        h = hashlib.md5((cost_type.__qualname__ + fm_type.__qualname__).encode()).hexdigest()[:7]
        suffix = f"{cls_.__name__}_{cost_type.__name__}_{fm_type.__name__}_{h}"
        config_name = f"config_{suffix}"

        class Cls(cls_):
            CostType = cost_type
            ForwardModelType = fm_type

            @dataclass
            class Config(configs.ConfigBase):
                model: cls_.ModelConfig = cls_.ModelConfig()
                cost: cost_type.Config = cost_type.Config()
                training: cls_.TrainingConfig = cls_.TrainingConfig()

        Cls.Config.__qualname__ = config_name
        Cls.Config.__name__ = config_name
        globals()[config_name] = Cls.Config
        return Cls

    return wrapper


@inject(cost_type=PolicyCost, fm_type=ForwardModel)
class MPURModule(pl.LightningModule):
    @dataclass
    class ModelConfig(configs.ModelConfig):
        forward_model_path: str = MISSING  # noqa: E501

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

    @dataclass
    class TrainingConfig(configs.TrainingConfig):
        pass

    def __init__(self, hparams=None):
        super().__init__()
        self.set_hparams(hparams)

        self.forward_model = self.ForwardModelType(self.config.model.forward_model_path, self.config.training.diffs)

        # exclude fm from the graph
        for p in self.forward_model.parameters():
            p.requires_grad = False

        self.policy_model = policy_models.DeterministicPolicy(
            n_cond=self.config.model.n_cond,
            n_feature=self.config.model.n_feature,
            n_actions=self.config.model.n_actions,
            h_height=self.config.model.h_height,
            h_width=self.config.model.h_width,
            n_hidden=self.config.model.n_hidden,
            diffs=self.config.training.diffs,
            turn_power=self.config.model.turn_power,
        )

        self.policy_model.train()

        if self.config.model.checkpoint is not None:
            checkpoint = torch.load(self.config.model.checkpoint)
            self.load_state_dict(checkpoint["state_dict"])

        if self.config.training.freeze_encoder:
            for name, p in self.policy_model.named_parameters():
                if "fc" not in name:
                    p.requires_grad = False

        self.augmenter = Augmenter(
            self.config.training.noise_augmentation_std, self.config.training.noise_augmentation_p,
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
            self.policy_model, batch, augmenter=self.augmenter, npred=self.config.model.n_pred
        )
        return predictions

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        predictions = self(batch)
        loss = self.policy_cost.calculate_cost(batch, predictions)
        logs = loss.copy()
        logs["action_norm"] = predictions["pred_actions"].norm(2, 2).pow(2).mean()
        res = loss["policy_loss"]
        for k in logs:
            self.log(
                "train/" + k, logs[k], on_step=True, logger=True, prog_bar=True,
            )

        # We retain the gradient of actions to later log it to wandb.
        predictions["pred_actions"].retain_grad()
        self.manual_backward(res, opt)
        self.log_action_grads(predictions["pred_actions"].grad)
        opt.step()

        return res

    def log_action_grads(self, grad):
        # Mean across all timesteps
        self.log("grads/all/action", grad.norm(2, dim=-1).mean(), on_step=False, on_epoch=True, logger=True)
        self.log("grads/all/action_acceleration", grad[..., 0].abs().mean(), on_step=False, on_epoch=True, logger=True)
        self.log("grads/all/action_turn", grad[..., 1].abs().mean(), on_step=False, on_epoch=True, logger=True)
        # The first timestep
        self.log("grads/0/action", grad[..., 0, :].norm(2, dim=-1).mean(), on_step=False, on_epoch=True, logger=True)
        self.log("grads/0/action_acceleration", grad[..., 0, 0].abs().mean(), on_step=False, on_epoch=True, logger=True)
        self.log("grads/0/action_turn", grad[..., 0, 1].abs().mean(), on_step=False, on_epoch=True, logger=True)

        # breakpoint()
        # self.wandb_log_grads_plot(grad.norm(dim=-1).mean(dim=0), "all")
        # self.wandb_log_grads_plot(grad[..., 0].abs().mean(dim=0), "acceleration")
        # self.wandb_log_grads_plot(grad[..., 1].abs().mean(dim=0), "turning")

    def wandb_log_grads_plot(self, vals, title):
        data = [[x, y] for (x, y) in enumerate(vals)]
        table = wandb.Table(data=data, columns=["x", "y"])
        self.logger.experiment.log({f"grads/plots/{title}": wandb.plot.line(table, "x", "y", title=title)})

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = self.policy_cost.calculate_cost(batch, predictions)
        res = loss["policy_loss"]
        for k in loss:
            self.log(
                "val/" + k, loss[k], on_epoch=True, logger=True,
            )
        return res

    @property
    def sample_step(self):
        return (
            self.trainer.global_step
            * self.config.training.batch_size
            * self.config.training.num_nodes
            * self.config.training.gpus
        )

    def validation_epoch_end(self, res):
        self.log("sample_step", self.sample_step)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.policy_model.parameters(),
            self.config.training.learning_rate
            * self.config.training.gpus
            * self.config.training.num_nodes
            * (6 / self.config.training.batch_size),
        )
        if self.config.training.scheduler is not None:
            if self.config.training.scheduler == "step":
                # we want to have 0.1 learning rate after 70% of training
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=int(self.config.training.n_epochs * 0.7), gamma=0.1,
                )
            elif self.config.training.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=int(self.config.training.n_epochs / 5)
                )

            return [optimizer], [scheduler]
        else:
            return optimizer

    def on_train_start(self):
        """
        The setup below happens after everything was moved
        to the correct device, and after the data was loaded.
        """
        self._setup_normalizer()
        self._setup_forward_model()
        self._setup_mixout()
        self._setup_policy_cost()
        self._setup_episode_evaluator()

    def _setup_normalizer(self):
        self.normalizer = Normalizer(self.trainer.datamodule.data_store.stats)
        self.policy_model.normalizer = self.normalizer
        if hasattr(self.forward_model, 'state_predictor'):
            self.forward_model.state_predictor.normalizer = self.normalizer

    def _setup_forward_model(self):
        self.forward_model.eval()
        self.forward_model.device = self.device
        self.forward_model.to(self.device)

    def _setup_policy_cost(self):
        self.policy_cost = self.CostType(self.config.cost, self.forward_model, self.normalizer)
        self.policy_cost.estimate_uncertainty_stats(self.train_dataloader())

    def _setup_mixout(self):
        if self.config.training.mixout_p is not None:
            # self.policy_model = MixoutDeterministicPolicy(
            #     self.policy_model, p=self.config.training.mixout_p
            # )
            self.mixout_wrapper = MixoutWrapper(self.config.training.mixout_p)
            self.policy_model = self.policy_model.apply(lambda x: self.mixout_wrapper(x))

    def _setup_episode_evaluator(self):
        self.eval_dataset = EvaluationDataset.from_data_store(
            self.trainer.datamodule.data_store, split="val", size_cap=25
        )
        self.evaluator = PolicyEvaluator(
            self.eval_dataset, num_processes=5, build_gradients=False, return_episode_data=False, enable_logging=False,
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
        model_type: str = "continuous_v2"


@inject(cost_type=PolicyCost, fm_type=ForwardModelV2)
class MPURVanillaV2Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type: str = "vanilla_v2"


@inject(cost_type=PolicyCost, fm_type=ForwardModelV3)
class MPURVanillaV3Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type: str = "vanilla_v3"


@inject(cost_type=PolicyCostContinuous, fm_type=ForwardModelV3)
class MPURContinuousV3Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type: str = "continuous_v3"

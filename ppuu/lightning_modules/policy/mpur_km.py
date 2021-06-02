"""Train a policy / controller"""
from dataclasses import dataclass

import torch

from ppuu.costs.policy_costs_km import PolicyCostKMTaper
from ppuu.lightning_modules.policy.mpur import MPURModule, inject
from ppuu.modeling.policy.mpc import MPCKMPolicy
from ppuu.wrappers import ForwardModelKM
from ppuu.modeling.forward_models_km_no_action import FwdCNNKMNoAction_VAE
from ppuu.modeling.forward_models import FwdCNN_VAE
from ppuu.data.entities import DatasetSample


@inject(cost_type=PolicyCostKMTaper, fm_type=ForwardModelKM)
class MPURKMTaperModule(MPURModule):
    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold_km(
            self.policy_model,
            batch,
            augmenter=self.augmenter,
            npred=self.config.model.n_pred,
        )
        return predictions


@inject(cost_type=PolicyCostKMTaper, fm_type=FwdCNNKMNoAction_VAE)
class MPURKMTaperV3Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type: str = "km_taper_v3"
        forward_model_path: str = (
            "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/"
            "fm_km_no_action_diff_64_even_lower_lr/seed=42/checkpoints/last.ckpt"
        )

    def build_log_dict(
        self, predictions: FwdCNN_VAE.Unfolding, cost: PolicyCostKMTaper.Cost
    ) -> dict:
        """Builds a dictionary of values to be logged from
        predictions and costs."""
        # we first use superclass' method, then add km cost specific stuff.
        result = super().build_log_dict(predictions, cost)
        result.update(
            {
                "reference_distance": cost.reference_distance.mean(),
                "speed": cost.speed.mean(),
                "destination": cost.destination.mean(),
            }
        )
        return result

    def calculate_cost(
        self, batch: DatasetSample, predictions: FwdCNN_VAE.Unfolding
    ) -> PolicyCostKMTaper.Cost:
        return self.policy_cost.calculate_cost(
            batch.conditional_state_seq,
            predictions.state_seq.states,
            predictions.actions,
            predictions.state_seq,
        )


@inject(cost_type=PolicyCostKMTaper, fm_type=FwdCNNKMNoAction_VAE)
class MPURKMTaperV3Module_TargetProp(MPURKMTaperV3Module):
    @dataclass
    class ModelConfig(MPURKMTaperV3Module.ModelConfig):
        model_type: str = "km_taper_v3_target_prop"
        mpc: MPCKMPolicy.Config = MPCKMPolicy.Config()
        mpc_cost: PolicyCostKMTaper.Config = PolicyCostKMTaper.Config()

    def _setup_mpc(self):
        self.mpc_cost = PolicyCostKMTaper(
            self.config.model.mpc_cost, self.forward_model, self.normalizer
        )
        self.mpc_cost.config.u_reg = 0
        self.mpc = MPCKMPolicy(
            self.forward_model,
            self.mpc_cost,
            self.normalizer,
            self.config.model.mpc,
        )

    def on_train_start(self):
        super().on_train_start()
        self._setup_mpc()

    def training_step(self, batch, batch_idx):
        batch = DatasetSample.from_tuple(batch)
        opt = self.optimizers()
        predictions = self(batch)
        print(
            f"fm predictions with policy are {predictions.state_seq.images.shape=}"
            f"{predictions.state_seq.states.shape=}"
        )

        with torch.no_grad():
            self.mpc.reset()
            mpc_actions = self.mpc(
                batch.conditional_state_seq,
                gt_future=predictions.state_seq,
                full_plan=True,
            )
            loss = self.calculate_cost(batch, predictions)
            logged_losses = self.build_log_dict(predictions, loss)

        for k, v in logged_losses.items():
            self.log(
                "train/" + k,
                v,
                on_step=True,
                logger=True,
                prog_bar=True,
            )

        assert predictions.actions.shape == mpc_actions.shape, (
            f"expected policy actions and mpc actions to be of the same shape,"
            f"got {predictions.actions.shape=} and {mpc_actions.shape=}"
        )
        action_diff = predictions.actions - mpc_actions
        action_diff = self.policy_cost.apply_gamma(action_diff)

        # Sum across time and actions dimensions, mean over batch.
        action_diff = action_diff.sum(dim=-1).sum(dim=-1).mean()

        # We retain the gradient of actions to later log it to wandb.
        predictions.actions.retain_grad()
        self.manual_backward(action_diff, optimizer=opt.optimizer)
        self.log_action_grads(predictions.actions.grad)
        opt.step()

        return action_diff


@inject(cost_type=PolicyCostKMTaper, fm_type=FwdCNNKMNoAction_VAE)
class GTMPURKMTaperV3Module(MPURModule):
    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold(
            self.policy_model,
            batch,
            augmenter=self.augmenter,
            npred=self.config.model.n_pred,
        )
        return predictions

    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type: str = "gt_km_taper_v3"
        forward_model_path: str = (
            "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/"
            "fm_km_no_action_diff_64_even_lower_lr/seed=42/checkpoints/last.ckpt"
        )

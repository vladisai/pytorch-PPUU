"""Train a policy / controller"""

from dataclasses import dataclass
import torch
import torch.optim as optim

from ppuu.costs.policy_costs_continuous import PolicyCostContinuous
from ppuu.lightning_modules.policy.mpur import (
    MPURModule,
    inject,
    ForwardModelV2,
)


@inject(cost_type=PolicyCostContinuous)
class MPURDreamingModule(MPURModule):
    @dataclass
    class TrainingConfig(MPURModule.TrainingConfig):
        lrt_z: float = 0.1
        n_z_updates: int = 2
        adversarial_frequency: int = 10
        n_adversarial_policy_updates: int = 1
        init_z_with_zero: bool = False

    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.adversarial_z = None

    def get_adversarial_z(self, batch):
        if self.config.training_config.init_z_with_zero:
            z = torch.zeros(
                self.config.model_config.n_pred
                * self.config.training_config.batch_size,
                32,
            ).to(self.device)
        else:
            z = self.forward_model.sample_z(
                self.config.model_config.n_pred
                * self.config.training_config.batch_size
            ).to(self.device)
        z = z.view(
            self.config.training_config.batch_size,
            self.config.model_config.n_pred,
            -1,
        ).detach()
        original_z = z.clone()
        z.requires_grad = True
        optimizer_z = self.get_z_optimizer(z)

        for i in range(self.config.training_config.n_z_updates):
            predictions = self.forward_model.unfold(
                self.policy_model, batch, z
            )
            cost, components = self.policy_cost.calculate_z_cost(
                batch, predictions, original_z
            )
            self.log_z(cost, components, "adv")
            optimizer_z.zero_grad()
            cost.backward()
            optimizer_z.step()

        mean_norm = (
            lambda x: x.reshape(-1, 32)
            .norm(dim=1)
            .pow(2)
            .div(32)
            .mean()
            .item()
        )

        mean_difference = mean_norm(z - original_z)
        mean_z_norm = mean_norm(z)
        mean_orig_norm = mean_norm(original_z)
        self.logger.log_custom(
            "z_difference", mean_difference, self.global_step
        )
        self.logger.log_custom("z_norm", mean_z_norm, self.global_step)
        self.logger.log_custom(
            "z_original_norm", mean_orig_norm, self.global_step
        )
        return z

    def log_z(self, cost, components, t):
        if hasattr(self.logger, "log_custom"):
            self.logger.log_custom(
                "z_cost", (cost.item(), t), self.global_step
            )
            self.logger.log_custom(
                "z_cost_proximity",
                (components["proximity_loss"].item(), t),
                self.global_step,
            )
            self.logger.log_custom(
                "z_cost_uncertainty",
                (components["u_loss"].item(), t),
                self.global_step,
            )

    def get_z_optimizer(self, Z):
        return torch.optim.Adam([Z], self.config.training_config.lrt_z)

    def forward_adversarial(self, batch):
        self.forward_model.eval()
        if self.adversarial_z is None:
            self.adversarial_z = self.get_adversarial_z(batch)
        predictions = self.forward_model.unfold(
            self.policy_model, batch, self.adversarial_z
        )
        return predictions

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs
    ):
        optimizer.step()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if batch_idx % self.config.training_config.adversarial_frequency == 0:
            predictions = self.forward_adversarial(batch)
            cost, components = self.policy_cost.calculate_z_cost(
                batch, predictions
            )
            self.log_z(cost, components, "adv")
        else:
            self.adversarial_z = None
            if optimizer_idx != 0:
                # We don't use extra optimizers unless we're doing adversarial
                # z.
                return {"loss": torch.tensor(0.0, requires_grad=True)}
            predictions = self(batch)
            cost, components = self.policy_cost.calculate_z_cost(
                batch, predictions
            )
            self.log_z(cost, components, "normal")
        loss = self.policy_cost.calculate_cost(batch, predictions)
        return {
            "loss": loss["policy_loss"],
            "log": loss,
            "progress_bar": loss,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.policy_model.parameters(),
            self.config.training_config.learning_rate,
        )
        return [
            optimizer
        ] * self.config.training_config.n_adversarial_policy_updates


@inject(cost_type=PolicyCostContinuous)
class MPURDreamingLBFGSModule(MPURDreamingModule):
    def get_adversarial_z(self, batch):
        z = self.forward_model.sample_z(
            self.config.model_config.n_pred
            * self.config.training_config.batch_size
        )
        z = z.view(
            self.config.training_config.batch_size,
            self.config.model_config.n_pred,
            -1,
        ).detach()
        original_z = z.clone()
        z.requires_grad = True
        optimizer_z = self.get_z_optimizer(z)

        def lbfgs_closure():
            optimizer_z.zero_grad()
            predictions = self.forward_model.unfold(
                self.policy_model, batch, z
            )
            cost, components = self.policy_cost.calculate_z_cost(
                batch, predictions
            )
            self.log_z(cost, components, "adv")
            optimizer_z.zero_grad()
            cost.backward()
            return cost

        optimizer_z.step(lbfgs_closure)
        self.logger.log_custom(
            "z_difference", (z - original_z).norm().item(), self.global_step
        )
        self.logger.log_custom("z_norm", (z).norm().item(), self.global_step)
        self.logger.log_custom(
            "z_original_norm", (original_z).norm().item(), self.global_step
        )
        return z

    def get_z_optimizer(self, Z):
        return torch.optim.LBFGS(
            [Z], max_iter=self.config.training_config.n_z_updates
        )


@inject(cost_type=PolicyCostContinuous, fm_type=ForwardModelV2)
class MPURDreamingV2Module(MPURDreamingModule):
    @dataclass
    class TrainingConfig(MPURDreamingModule.TrainingConfig):
        def auto_batch_size(self):
            if self.batch_size == -1:
                gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.batch_size = int((gpu_gb / 8) * 6)
                print("auto batch size is set to", self.batch_size)
            self.auto_n_epochs()

    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        forward_model_path: str = "/home/us441/nvidia-collab/vlad/results/refactored_debug/fm_km_5_states_resume_lower_lr/seed=42/checkpoints/epoch=23_success_rate=0.ckpt"

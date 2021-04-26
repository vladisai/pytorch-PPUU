"""Train a policy / controller"""
from dataclasses import dataclass

import torch

from ppuu.costs.policy_costs_km import PolicyCostKM, PolicyCostKMTaper
from ppuu.data.dataloader import overlay_ego_car
from ppuu.lightning_modules.policy.mpur import (
    ForwardModelV3,
    MPURModule,
    inject,
)
from ppuu.modeling.mpc import MPCKMPolicy
from ppuu.wrappers import ForwardModelKM


@inject(cost_type=PolicyCostKM, fm_type=ForwardModelKM)
class MPURKMModule(MPURModule):
    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold_km(
            self.policy_model,
            batch,
            augmenter=self.augmenter,
            npred=self.config.model.n_pred,
        )
        return predictions


@inject(cost_type=PolicyCostKMTaper, fm_type=ForwardModelKM)
class MPURKMTaperModule(MPURKMModule):
    pass


@inject(cost_type=PolicyCostKMTaper, fm_type=ForwardModelV3)
class MPURKMTaperV3Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type: str = "km_taper_v3"
        forward_model_path: str = (
            "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/"
            "fm_km_no_action_diff_64_even_lower_lr/seed=42/checkpoints/last.ckpt"
        )


@inject(cost_type=PolicyCostKMTaper, fm_type=ForwardModelV3)
class MPURKMTaperV3Module_TargetProp(MPURKMTaperV3Module):
    @dataclass
    class ModelConfig(MPURKMTaperV3Module.ModelConfig):
        model_type: str = "km_taper_v3_target_prop"
        mpc: MPCKMPolicy.Config = MPCKMPolicy.Config()
        mpc_cost: PolicyCostKMTaper.Config = PolicyCostKMTaper.Config()

    def on_train_start(self):
        super().on_train_start()
        self.mpc_cost = PolicyCostKMTaper(
            self.config.model.mpc_cost, None, self.normalizer
        )
        self.mpc = MPCKMPolicy(
            self.forward_model,
            self.mpc_cost,
            self.normalizer,
            self.config.model.mpc,
        )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        predictions = self(batch)
        input_images = overlay_ego_car(
            batch["input_images"], batch["ego_cars"]
        )
        metadata = {}
        mpc_actions = self.mpc.batched(  # noqa
            input_images,
            batch["input_states"],
            car_size=batch["car_sizes"],
            init=predictions["pred_actions"],
            pred_images=predictions["pred_images"],
            pred_states=predictions["pred_states"],
            metadata=metadata,
        )
        print(metadata["costs"])

        loss = self.policy_cost.calculate_cost(batch, predictions)
        loss["action_norm"] = (
            predictions["pred_actions"].norm(2, 2).pow(2).mean()
        )
        res = loss["policy_loss"].mean()
        for k in loss:
            v = loss[k]
            if torch.is_tensor(v):
                v = v.mean()
            if v is not None:
                self.log(
                    "train/" + k,
                    v,
                    on_step=True,
                    logger=True,
                    prog_bar=True,
                )

        # We retain the gradient of actions to later log it to wandb.
        predictions["pred_actions"].retain_grad()
        self.manual_backward(res, optimizer=opt.optimizer)
        self.log_action_grads(predictions["pred_actions"].grad)
        opt.step()

        return res


@inject(cost_type=PolicyCostKMTaper, fm_type=ForwardModelV3)
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

"""Train a policy / controller"""

from dataclasses import dataclass

from ppuu.lightning_modules.policy.mpur import (
    MPURModule,
    inject,
    ForwardModelV3,
)
from ppuu.costs.policy_costs_km import (
    PolicyCostKM,
    PolicyCostKMSplit,
    PolicyCostKMTaper,
)
from ppuu.wrappers import ForwardModelKM


@inject(cost_type=PolicyCostKM, fm_type=ForwardModelKM)
class MPURKMModule(MPURModule):
    def forward(self, batch):
        self.forward_model.eval()
        predictions = self.forward_model.unfold_km(
            self.policy_model, batch, augmenter=self.augmenter
        )
        return predictions


@inject(cost_type=PolicyCostKMSplit, fm_type=ForwardModelKM)
class MPURKMSplitModule(MPURKMModule):
    pass


@inject(cost_type=PolicyCostKMTaper, fm_type=ForwardModelKM)
class MPURKMTaperModule(MPURKMModule):
    pass


@inject(cost_type=PolicyCostKMTaper, fm_type=ForwardModelV3)
class MPURKMTaperV3Module(MPURModule):
    @dataclass
    class ModelConfig(MPURModule.ModelConfig):
        model_type : str = 'km_taper_v3'
        forward_model_path: str = "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/fm_km_no_action_diff_64_even_lower_lr/seed=42/checkpoints/last.ckpt"

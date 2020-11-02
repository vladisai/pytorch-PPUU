"""Train a policy / controller"""


from ppuu.lightning_modules.policy.mpur import MPURModule, inject
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

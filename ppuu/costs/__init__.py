from ppuu.costs.policy_costs import PolicyCost
from ppuu.costs.policy_costs_continuous import PolicyCostContinuous
from ppuu.costs.policy_costs_km import (
    PolicyCostKM,
    PolicyCostKMSplit,
    PolicyCostKMTaper,
)

MODEL_MAPPING = dict(
    vanilla=PolicyCost,
    continuous=PolicyCostContinuous,
    km=PolicyCostKM,
    km_split=PolicyCostKMSplit,
    km_taper=PolicyCostKMTaper,
)


def get_cost_model_from_name(name):
    return MODEL_MAPPING[name]

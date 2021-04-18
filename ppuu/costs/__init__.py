import argparse

from ppuu.costs.policy_costs import PolicyCost
from ppuu.costs.policy_costs_continuous import PolicyCostContinuous
from ppuu.costs.policy_costs_km import PolicyCostKMTaper

MODEL_MAPPING = dict(
    vanilla=PolicyCost,
    continuous=PolicyCostContinuous,
    km_taper=PolicyCostKMTaper,
)


def get_cost_model_from_name(name):
    return MODEL_MAPPING[name]


def get_module_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cost_type",
        type=str,
        help="Pick the cost type to run",
        required=True,
    )
    args, _ = parser.parse_known_args()
    return get_cost_model_from_name(args.model_type)

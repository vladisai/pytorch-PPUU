import argparse

from ppuu.lightning_modules.policy.mpur import (
    MPURContinuousModule,
    MPURContinuousV3Module,
    MPURModule,
    MPURVanillaV3Module,
)
from ppuu.lightning_modules.policy.mpur_dreaming import (
    MPURDreamingLBFGSModule,
    MPURDreamingModule,
    MPURDreamingV3Module,
)
from ppuu.lightning_modules.policy.mpur_km import (
    MPURKMTaperModule,
    MPURKMTaperV3Module,
    MPURKMTaperV3Module_TargetProp,
    MPURKMTaperV3Module_TargetPropOneByOne,
)

MODULES_DICT = dict(
    vanilla=MPURModule,
    dreaming=MPURDreamingModule,
    dreaming_lbfgs=MPURDreamingLBFGSModule,
    km_taper=MPURKMTaperModule,
    km_taper_v3=MPURKMTaperV3Module,
    continuous=MPURContinuousModule,
    vanilla_v3=MPURVanillaV3Module,
    continuous_v3=MPURContinuousV3Module,
    dreaming_v3=MPURDreamingV3Module,
    km_taper_v3_target_prop=MPURKMTaperV3Module_TargetProp,
    km_taper_v3_target_prop_one_by_one=MPURKMTaperV3Module_TargetPropOneByOne,
)


def get_module_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        help="Pick the model type to run",
        required=True,
    )
    args, _ = parser.parse_known_args()
    return get_module(args.model_type)


def get_module(name):
    return MODULES_DICT[name]

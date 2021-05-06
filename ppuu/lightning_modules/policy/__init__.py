import argparse

from ppuu.lightning_modules.policy.mpur import (
    MPURContinuousModule,
    MPURContinuousV2Module,
    MPURContinuousV3Module,
    MPURModule,
    MPURVanillaV2Module,
    MPURVanillaV3Module,
)
from ppuu.lightning_modules.policy.mpur_dreaming import (
    MPURDreamingLBFGSModule,
    MPURDreamingModule,
    MPURDreamingV2Module,
    MPURDreamingV3Module,
)
from ppuu.lightning_modules.policy.mpur_km import (
    MPURKMTaperModule,
    MPURKMTaperV3Module,
    MPURKMTaperV3Module_TargetProp,
)

MODULES_DICT = dict(
    vanilla=MPURModule,
    dreaming=MPURDreamingModule,
    dreaming_lbfgs=MPURDreamingLBFGSModule,
    km_taper=MPURKMTaperModule,
    km_taper_v3=MPURKMTaperV3Module,
    continuous=MPURContinuousModule,
    continuous_v2=MPURContinuousV2Module,
    vanilla_v2=MPURVanillaV2Module,
    dreaming_v2=MPURDreamingV2Module,
    vanilla_v3=MPURVanillaV3Module,
    continuous_v3=MPURContinuousV3Module,
    dreaming_v3=MPURDreamingV3Module,
    km_taper_v3_target_prop=MPURKMTaperV3Module_TargetProp,
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

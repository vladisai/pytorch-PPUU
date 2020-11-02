import argparse

from ppuu.lightning_modules.policy.mpur import (
    MPURModule,
    MPURVanillaV2Module,
    MPURVanillaV3Module,
    MPURContinuousModule,
    MPURContinuousV2Module,
    MPURContinuousV3Module,
)
from ppuu.lightning_modules.policy.mpur_dreaming import (
    MPURDreamingModule,
    MPURDreamingLBFGSModule,
    MPURDreamingV2Module,
)
from ppuu.lightning_modules.policy.mpur_km import (
    MPURKMModule,
    MPURKMSplitModule,
    MPURKMTaperModule,
)


MODULES_DICT = dict(
    vanilla=MPURModule,
    dreaming=MPURDreamingModule,
    dreaming_lbfgs=MPURDreamingLBFGSModule,
    km=MPURKMModule,
    km_split=MPURKMSplitModule,
    km_taper=MPURKMTaperModule,
    continuous=MPURContinuousModule,
    continuous_v2=MPURContinuousV2Module,
    vanilla_v2=MPURVanillaV2Module,
    dreaming_v2=MPURDreamingV2Module,
    vanilla_v3=MPURVanillaV3Module,
    continuous_v3=MPURContinuousV3Module,
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

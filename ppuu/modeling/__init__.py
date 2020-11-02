from ppuu.modeling.forward_models import FwdCNN, FwdCNN_VAE
from ppuu.modeling.forward_models_km import FwdCNNKM, FwdCNNKM_VAE
from ppuu.modeling.forward_models_km_no_action import (
    FwdCNNKMNoAction,
    FwdCNNKMNoAction_VAE,
)

FM_MAPPING = {
    "vanilla": FwdCNN_VAE,
    "km": FwdCNNKM_VAE,
    "km_no_action": FwdCNNKMNoAction_VAE,
}


def get_fm(name: str):
    return FM_MAPPING[name]

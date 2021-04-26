from ppuu.modeling.forward_models import FwdCNN_VAE


def get_fm():
    return FwdCNN_VAE(
        layers=3,
        nfeature=256,
        dropout=0.1,
        h_height=14,
        h_width=3,
        height=117,
        width=24,
        n_actions=2,
        hidden_size=256 * 14 * 3,
        ncond=20,
        predict_state=True,
        nz=32,
        enable_kld=True,
        enable_latent=True,
    )

import unittest

from ppuu.data.dataloader import Normalizer
from ppuu.modeling.forward_models import FwdCNN_VAE
from ppuu.modeling.forward_models_km import FwdCNNKM_VAE
from ppuu.modeling.forward_models_km_no_action import FwdCNNKMNoAction_VAE
from ppuu.modeling.km import StatePredictor
from ppuu.test.mock_dataset import get_mock_dataloader


class TestFMModels(unittest.TestCase):
    def run_forward_unfold(self, m):
        dl = get_mock_dataloader()
        batch = next(iter(dl))

        m.forward(
            batch.conditional_state_seq,
            batch.target_action_seq,
            batch.target_state_seq,
        )

        m.unfold(
            batch.conditional_state_seq,
            batch.target_action_seq,
        )

    def test_FwdCNN_VAE(self):
        m = FwdCNN_VAE(
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
        self.run_forward_unfold(m)
        m.enable_latent = False
        self.run_forward_unfold(m)

    def test_FwdCNNKM_VAE(self):
        m = FwdCNNKM_VAE(
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
            state_predictor=StatePredictor(
                diff=False, normalizer=Normalizer.dummy()
            ),
            nz=32,
            enable_kld=True,
            enable_latent=True,
        )
        self.run_forward_unfold(m)
        m.enable_latent = False
        self.run_forward_unfold(m)

    def test_FwdCNNKMNoAction_VAE(self):
        m = FwdCNNKMNoAction_VAE(
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
            state_predictor=StatePredictor(
                diff=False, normalizer=Normalizer.dummy()
            ),
            nz=32,
            enable_kld=True,
            enable_latent=True,
        )
        self.run_forward_unfold(m)
        m.enable_latent = False
        self.run_forward_unfold(m)

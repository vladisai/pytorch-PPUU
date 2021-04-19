import unittest

from ppuu.costs import PolicyCost, PolicyCostContinuous, PolicyCostKMTaper
from ppuu.data.dataloader import Normalizer
from ppuu.modeling.forward_models import FwdCNN_VAE
from ppuu.test.mock_dataset import get_mock_dataloader


class TestFMModels(unittest.TestCase):
    def setUp(self):
        self.fm = self.get_fm()
        self.fm = self.fm.cuda()

    def get_fm(self):
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

    def run_cost_test(self, c):
        dl = get_mock_dataloader()
        batch = next(iter(dl))
        batch = batch.cuda()

        c.estimate_uncertainty_stats(dl)

        c.calculate_cost(
            batch.conditional_state_seq,
            batch.target_action_seq,
            batch.target_state_seq,
        )

    def test_vanilla_cost(self):
        cost_config = PolicyCost.Config()
        cost_config.uncertainty_n_models = 2
        cost_config.uncertainty_n_batches = 2
        c = PolicyCost(cost_config, self.fm, Normalizer.dummy())
        self.run_cost_test(c)

    def test_continuous_cost(self):
        cost_config = PolicyCostContinuous.Config()
        cost_config.uncertainty_n_models = 2
        cost_config.uncertainty_n_batches = 2
        cost_config.skip_contours = False

        for skip_contours in [True, False]:
            cost_config.skip_contours = skip_contours
            c = PolicyCostContinuous(cost_config, self.fm, Normalizer.dummy())
            self.run_cost_test(c)

    def test_km_taper(self):
        cost_config = PolicyCostKMTaper.Config()
        cost_config.uncertainty_n_models = 2
        cost_config.uncertainty_n_batches = 2
        cost_config.skip_contours = False

        c = PolicyCostKMTaper(cost_config, self.fm, Normalizer.dummy())

        dl = get_mock_dataloader()
        batch = next(iter(dl))
        batch = batch.cuda()

        c.estimate_uncertainty_stats(dl)

        c.calculate_cost(
            batch.conditional_state_seq,
            batch.target_state_seq.states,
            batch.target_action_seq,
            batch.target_state_seq,
        )

import unittest

from ppuu.costs import PolicyCostKMTaper
from ppuu.data import dataloader
from ppuu.modeling.policy.mpc import MPCKMPolicy
from ppuu.test import utils
from ppuu.test.mock_dataset import get_mock_dataloader


class TestKMMPC(unittest.TestCase):
    def setUp(self):
        self.fm = utils.get_fm()
        self.fm = self.fm.cuda()

    def test_init(self):
        normalizer = dataloader.Normalizer.dummy()

        cost_config = PolicyCostKMTaper.Config()
        cost_config.uncertainty_n_models = 2
        cost_config.uncertainty_n_batches = 2
        cost_config.skip_contours = False
        cost_config.u_reg = 0

        mpc_config = MPCKMPolicy.Config()
        mpc_config.fm_unfold_samples = 4
        mpc_config.batch_size = 3

        cost = PolicyCostKMTaper(cost_config, self.fm, normalizer)
        policy = MPCKMPolicy(self.fm, cost, normalizer, mpc_config, None)

        dl = get_mock_dataloader()
        sample = next(iter(dl))

        # we want to call the policy to output some stuff.
        actions = sample.target_action_seq.unsqueeze(1).repeat_interleave(
            3, dim=1
        )
        target_state_seq = sample.target_state_seq.map(
            lambda x: x.unsqueeze(1).repeat_interleave(4, dim=1)
        )

        x = policy.get_cost(
            sample.conditional_state_seq.cuda(),
            actions.cuda(),
            target_state_seq.cuda(),
        )
        assert list(x.shape) == [2, 3]

        policy(sample.conditional_state_seq.cuda())

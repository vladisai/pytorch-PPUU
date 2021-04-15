import tempfile
import unittest

from omegaconf import OmegaConf

from ppuu import eval_policy, train_fm, train_policy
from ppuu.lightning_modules.fm import FM
from ppuu.lightning_modules.policy import MPURVanillaV3Module


class TestTrainFM(unittest.TestCase):
    CONFIG_PATHS = ["test_dataset.yml"]

    def load_file_configs(self):
        configs = []
        for path in self.CONFIG_PATHS:
            configs.append(OmegaConf.load(path))
        omega_config = OmegaConf.merge(*configs)
        return omega_config

    def test_train_fm(self):
        with tempfile.TemporaryDirectory(prefix="ppuu_fm_test_") as tempdir:
            config = FM.Config()
            # config.training.fast_dev_run = True
            file_config = self.load_file_configs()
            config.training.dataset = file_config.training.dataset
            config.training.experiment_name = "debug"
            config.training.output_dir = tempdir
            config.training.checkpoints = 100
            config.training.wandb_offline = True
            config.training.n_epochs = 2
            config.training.epoch_size = 2
            config.training.n_steps = 64 * 4
            config.training.batch_size = 64
            config.training.validation_size = 2
            config.training.validation_period = 1

            train_fm.main(config)
        print("done")


class TestTrainPolicy(unittest.TestCase):
    CONFIG_PATHS = ["test_dataset.yml", "test_fm_checkpoint.yml"]

    def load_file_configs(self):
        configs = []
        for path in self.CONFIG_PATHS:
            configs.append(OmegaConf.load(path))
        omega_config = OmegaConf.merge(*configs)
        return omega_config

    def test_train_policy(self):
        with tempfile.TemporaryDirectory(
            prefix="ppuu_policy_test_"
        ) as tempdir:
            config = MPURVanillaV3Module.Config()
            # config.training.fast_dev_run = True
            file_config = self.load_file_configs()
            config.training.dataset = file_config.training.dataset
            config.model.forward_model_path = (
                file_config.model.forward_model_path
            )
            config.training.diffs = file_config.training.diffs
            config.training.experiment_name = "debug"
            config.training.output_dir = tempdir
            config.training.checkpoints = 100
            config.training.wandb_offline = True
            config.training.n_epochs = 2
            config.training.epoch_size = 2
            config.training.n_steps = 6 * 4
            config.training.batch_size = 6
            config.training.validation_size = 2
            config.cost.uncertainty_n_batches = 2
            config.training.validation_period = 1

            train_policy.main(config)


class TestEvaluator(unittest.TestCase):
    CONFIG_PATHS = ["test_eval_dataset.yml", "test_policy_checkpoint.yml"]

    def load_file_configs(self):
        configs = []
        for path in self.CONFIG_PATHS:
            configs.append(OmegaConf.load(path))
        omega_config = OmegaConf.merge(*configs)
        return omega_config

    def test_eval_policy(self):
        with tempfile.TemporaryDirectory(prefix="ppuu_eval_test_") as tempdir:
            file_config = self.load_file_configs()
            config = eval_policy.EvalConfig(
                checkpoint_path=file_config.checkpoint_path,
                output_dir=tempdir,
                dataset=file_config.training.dataset,
                test_size_cap=2,
                num_processes=0,
            )
            eval_policy.main(config)

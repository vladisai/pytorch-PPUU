import argparse
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch.multiprocessing

from ppuu.dataloader import EvaluationDataset
from ppuu.eval import PolicyEvaluator
from ppuu.lightning_modules import MPURKMTaperModule as Module
from ppuu.train_policy import CustomLogger

EPOCHS = 21


def generate_config():
    config = Module.Config()
    config.cost_config.lambda_l = 10 ** np.random.uniform(-3, 1)
    config.cost_config.lambda_p = (
        config.cost_config.lambda_l * np.random.uniform(0.1, 2) ** 2
    )
    config.cost_config.lambda_o = (
        config.cost_config.lambda_l * np.random.uniform(0.1, 2) ** 2
    )
    config.cost_config.u_reg = (
        config.cost_config.lambda_l * np.random.uniform(0, 1) ** 2
    )
    config.cost_config.lambda_a = (
        config.cost_config.lambda_l * np.random.uniform(0, 1) ** 2
    )
    config.cost_config.lambda_j = (
        config.cost_config.lambda_l * np.random.uniform(0, 1) ** 2
    )
    if np.random.choice([True, False]):
        config.cost_config.agg_func_str = "sum"
    else:
        config.cost_config.agg_func_str = (
            f"logsumexp-{np.random.randint(15, 85)}"
        )
    config.cost_config.masks_power_x = np.random.uniform(1, 10)
    config.cost_config.masks_power_y = np.random.uniform(1, 10)
    config.model_config.model_type = "km_taper"
    config.training_config.noise_augmentation_std = 0.07
    config.training_config.noise_augmentation_p = 0.5
    return config


def generate_config_choice():
    config = Module.Config()
    config.cost_config.lambda_l = np.random.choice([0.01, 0.1, 1, 10])
    config.cost_config.lambda_p = (
        config.cost_config.lambda_l
        * np.random.choice([0.1, 0.5, 1, 2, 10, 100])
    )
    config.cost_config.lambda_o = (
        config.cost_config.lambda_l
        * np.random.choice([0.1, 0.5, 1, 2, 10, 100])
    )
    config.cost_config.u_reg = config.cost_config.lambda_l * np.random.choice(
        [0.05, 0.5, 1, 5, 10]
    )
    config.cost_config.lambda_a = (
        config.cost_config.lambda_l
        * np.random.choice([0.001, 0.01, 0.1, 1, 10])
    )
    config.cost_config.lambda_j = (
        config.cost_config.lambda_l
        * np.random.choice([0.001, 0.01, 0.1, 1, 10])
    )
    if np.random.choice([True, False]):
        config.cost_config.agg_func_str = "sum"
    else:
        config.cost_config.agg_func_str = (
            f"logsumexp-{np.random.choice([25, 50, 75])}"
        )
    config.cost_config.masks_power_x = np.random.choice([2, 3, 4, 5, 6])
    config.cost_config.masks_power_y = np.random.choice([2, 3, 4, 5, 6])
    config.model_config.model_type = "km_taper"

    config.training_config.learning_rate = np.random.choice(
        [1e-5, 1e-4, 1e-3, 1e-2]
    )

    return config


def run_trial(output_dir):
    config = generate_config()
    config.training_config.n_epochs = EPOCHS
    config.training_config.epoch_size = 500
    # config.training_config.n_epochs = 2
    # config.training_config.epoch_size = 10
    config.training_config.validation_size = 10
    config.training_config.validation_eval = False
    # config.training_config.dataset = configs.DATASET_PATHS_MAPPING[""]
    config.cost_config.uncertainty_n_batches = 100
    exp_name = f"grid_search_{time.time()}"

    for seed in [np.random.randint(1000)]:
        config.training_config.seed = seed
        logger = CustomLogger(
            save_dir=output_dir,
            name=exp_name,
            version=f"seed={config.training_config.seed}",
        )
        trainer = pl.Trainer(
            gpus=1,
            gradient_clip_val=50.0,
            max_epochs=config.training_config.n_epochs,
            check_val_every_n_epoch=EPOCHS,
            num_sanity_val_steps=0,
            checkpoint_callback=pl.callbacks.ModelCheckpoint(
                filepath=os.path.join(logger.log_dir, "checkpoints"),
                save_top_k=-1,
                save_last=True,
            ),
            logger=logger,
        )
        model = Module(config)
        trainer.fit(model)

        eval_dataset = EvaluationDataset.from_data_store(
            model.data_store, split="val", size_cap=200
        )
        evaluator = PolicyEvaluator(
            eval_dataset,
            num_processes=5,
            build_gradients=False,
            return_episode_data=False,
            enable_logging=False,
        )
        eval_results = evaluator.evaluate(model)
        logger.log_custom(
            "success_rate", eval_results["stats"]["success_rate"]
        )
        logger.save()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, required=True, help="output dir"
    )
    args = parser.parse_args()

    for i in range(1000):
        run_trial(args.output_dir)

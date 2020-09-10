import time
import os
import argparse
import dataclasses

import numpy as np

import pytorch_lightning as pl
import torch.multiprocessing

import wandb

from ppuu.lightning_modules import MPURContinuousModule as Module
from ppuu.train_utils import CustomLogger
from ppuu.data.dataloader import EvaluationDataset
from ppuu.eval import PolicyEvaluator
from ppuu.data import NGSIMDataModule

from ppuu import configs


EPOCHS = 21


def run_trial(config, wandb_obj):
    config.training_config.n_epochs = EPOCHS
    config.training_config.epoch_size = 500
    config.training_config.validation_size = 10
    config.training_config.validation_eval = False
    config.cost_config.uncertainty_n_batches = 100

    # config.training_config.dataset = configs.DATASET_PATHS_MAPPING["50"]
    # config.training_config.n_epochs = 1
    # config.training_config.epoch_size = 2

    exp_name = f"grid_search_{time.time()}"

    for seed in [np.random.randint(1000)]:
        config.training_config.seed = seed
        logger = CustomLogger(
            save_dir=config.training_config.output_dir,
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

        datamodule = NGSIMDataModule(
            config.training_config.dataset,
            config.training_config.epoch_size,
            config.training_config.validation_size,
            config.training_config.batch_size,
        )

        pl.seed_everything(config.training_config.seed)

        trainer.fit(model, datamodule)

        eval_dataset = EvaluationDataset.from_data_store(
            datamodule.data_store, split="val", size_cap=200
        )
        model = model.eval()
        evaluator = PolicyEvaluator(
            eval_dataset,
            num_processes=0,
            build_gradients=False,
            return_episode_data=False,
            enable_logging=False,
        )
        eval_results = evaluator.evaluate(model)
        logger.save()
        return eval_results["stats"]["success_rate"]


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    # Set up your default hyperparameters before wandb.init
    # so they get properly set in the sweep
    hyperparameter_defaults = dataclasses.asdict(Module.Config())

    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    config = Module.Config.parse_from_dict(dict(wandb.config))
    if config.training_config.output_dir is None:
        config.training_config.output_dir = (
            "/home/us441/nvidia-collab/vlad/results/grid_continuous"
        )
    print(config)

    success_rate = run_trial(config, wandb)

    metrics = {"success_rate": success_rate}
    wandb.log(metrics)

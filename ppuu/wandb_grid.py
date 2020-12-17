import time
import os
import argparse
import dataclasses

import numpy as np

import pytorch_lightning as pl
import torch.multiprocessing

import wandb

from ppuu.lightning_modules.policy import MPURKMTaperV3Module as Module
from ppuu.train_utils import CustomLoggerWB
from ppuu.data.dataloader import EvaluationDataset
from ppuu.eval import PolicyEvaluator
from ppuu.data import NGSIMDataModule

from ppuu import configs


EPOCHS = 21


def run_trial(config, run):
    config.training.n_epochs = -1
    config.training.batch_size = -1
    config.training.n_steps = 2e5
    config.training.epoch_size = 500
    config.training.validation_size = 10
    config.training.validation_eval = False
    config.training.experiment_name = f"grid_search_{time.time()}"
    config.cost.uncertainty_n_batches = 100
    config.model.forward_model_path = "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/fm_km_no_action_64/seed=42/checkpoints/last.ckpt"

    config.training.auto_batch_size()

    print("config", config)

    for seed in [np.random.randint(1000)]:
        config.training.seed = seed
        logger = CustomLoggerWB(
            save_dir=config.training.output_dir,
            experiment_name=config.training.experiment_name,
            seed=str(config.training.seed),
            experiment=run,
        )

        n_checkpoints = 5
        if config.training.n_steps is not None:
            n_checkpoints = max(
                n_checkpoints, int(config.training.n_steps / 1e5)
            )

        period = max(1, config.training.n_epochs // n_checkpoints)

        trainer = pl.Trainer(
            gpus=config.training.gpus,
            num_nodes=config.training.num_nodes,
            gradient_clip_val=5.0,
            max_epochs=config.training.n_epochs,
            check_val_every_n_epoch=period,
            num_sanity_val_steps=0,
            fast_dev_run=config.training.fast_dev_run,
            distributed_backend=config.training.distributed_backend,
            checkpoint_callback=pl.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    logger.log_dir, "checkpoints", "{epoch}_{sample_step}"
                ),
                save_top_k=None,
                monitor=None,
            ),
            logger=logger,
            weights_save_path=logger.log_dir,
            track_grad_norm=False,
            automatic_optimization=False,
        )
        model = Module(config)

        datamodule = NGSIMDataModule(
            config.training.dataset,
            config.training.epoch_size,
            config.training.validation_size,
            config.training.batch_size,
            diffs=False,
        )

        pl.seed_everything(config.training.seed)

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
    hyperparameter_defaults = {}

    # Pass your defaults to wandb.init
    run = wandb.init(config=hyperparameter_defaults, reinit=True)
    c_dict = dict(wandb.config)

    # translate some params from log scale to normal scale
    log_params = ['lambda_a', 'lambda_j', 'u_reg', 'mask_coeff', 'learning_rate']
    for k in c_dict:
        if k in log_params:
            c_dict[k] = 10.0 ** c_dict[k]

    c_dict['masks_power_x'] = c_dict['powers']
    c_dict['masks_power_y'] = c_dict['powers']
    c_dict['lambda_l'] = c_dict['lambda_']
    c_dict['lambda_o'] = c_dict['lambda_']
    c_dict['lambda_p'] = c_dict['lambda_'] * 4

    del c_dict['powers']

    print(c_dict)

    config = Module.Config.parse_from_flat_dict(c_dict)
    if config.training.output_dir is None:
        config.training.output_dir = (
            "/home/us441/nvidia-collab/vlad/results/policy/grid_km"
        )

    success_rate = run_trial(config, run)

    metrics = {"success_rate": success_rate}
    wandb.log(metrics)

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
    config.training_config.n_epochs = -1
    config.training_config.batch_size = -1
    config.training_config.n_steps = 2e5
    config.training_config.epoch_size = 500
    config.training_config.validation_size = 10
    config.training_config.validation_eval = False
    config.cost_config.uncertainty_n_batches = 100
    config.model_config.forward_model_path = "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/fm_km_no_action_64/seed=42/checkpoints/last.ckpt"

    config.training_config.auto_batch_size()

    # config.training_config.dataset = configs.DATASET_PATHS_MAPPING["50"]
    # config.training_config.n_epochs = 1
    # config.training_config.epoch_size = 2

    exp_name = f"grid_search_{time.time()}"

    for seed in [np.random.randint(1000)]:
        config.training_config.seed = seed
        logger = CustomLoggerWB(
            save_dir=config.training_config.output_dir,
            experiment_name=exp_name,
            seed=str(config.training_config.seed),
            experiment=run,
        )

        n_checkpoints = 5
        if config.training_config.n_steps is not None:
            n_checkpoints = max(
                n_checkpoints, int(config.training_config.n_steps / 1e5)
            )

        period = max(1, config.training_config.n_epochs // n_checkpoints)

        trainer = pl.Trainer(
            gpus=config.training_config.gpus,
            num_nodes=config.training_config.num_nodes,
            gradient_clip_val=5.0,
            max_epochs=config.training_config.n_epochs,
            check_val_every_n_epoch=period,
            num_sanity_val_steps=0,
            fast_dev_run=config.training_config.fast_dev_run,
            distributed_backend=config.training_config.distributed_backend,
            checkpoint_callback=pl.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    logger.log_dir, "checkpoints", "{epoch}_{sample_step}"
                ),
                save_top_k=None,
                monitor=None,
            ),
            logger=logger,
            weights_save_path=logger.log_dir,
            terminate_on_nan=True,
            track_grad_norm=2,
        )
        model = Module(config)

        datamodule = NGSIMDataModule(
            config.training_config.dataset,
            config.training_config.epoch_size,
            config.training_config.validation_size,
            config.training_config.batch_size,
            diffs=False,
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
    if config.training_config.output_dir is None:
        config.training_config.output_dir = (
            "/home/us441/nvidia-collab/vlad/results/policy/grid_km"
        )
    print("config", config)

    success_rate = run_trial(config, run)

    metrics = {"success_rate": success_rate}
    wandb.log(metrics)

"""Train a policy / controller"""
import os

import pytorch_lightning as pl
import torch.multiprocessing

from ppuu import lightning_modules
from ppuu import slurm
from ppuu.data import NGSIMDataModule

from ppuu.train_utils import CustomLoggerWB, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor


def main(config):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    config.training_config.auto_batch_size()

    if config.training_config.debug or config.training_config.fast_dev_run:
        config.training_config.set_dataset("50")
        config.training_config.epoch_size = 10
        config.training_config.n_epochs = 10
        config.cost_config.uncertainty_n_batches = 10

    module = lightning_modules.policy.get_module(
        config.model_config.model_type
    )
    datamodule = NGSIMDataModule(
        config.training_config.dataset,
        config.training_config.epoch_size,
        config.training_config.validation_size,
        config.training_config.batch_size,
        workers=0,
        diffs=config.training_config.diffs,
    )

    pl.seed_everything(config.training_config.seed)

    logger = CustomLoggerWB(
        save_dir=config.training_config.output_dir,
        experiment_name=config.training_config.experiment_name,
        seed=f"seed={config.training_config.seed}",
        version=config.training_config.version,
        project="PPUU_policy",
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
        gradient_clip_val=50.0,
        max_epochs=config.training_config.n_epochs,
        check_val_every_n_epoch=period,
        num_sanity_val_steps=0,
        fast_dev_run=config.training_config.fast_dev_run,
        distributed_backend=config.training_config.distributed_backend,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                logger.log_dir, "checkpoints", "{epoch}_{sample_step}"
            ),
            save_top_k=-1,
            monitor=None,
        ),
        logger=logger,
        resume_from_checkpoint=config.training_config.resume_from_checkpoint,
        weights_save_path=logger.log_dir,
        automatic_optimization=False,
    )

    model = module(config)
    trainer.fit(model, datamodule=datamodule)
    return model


if __name__ == "__main__":
    module = lightning_modules.policy.get_module_from_command_line()
    config = module.Config.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor(
            job_name=config.training_config.experiment_name,
            cpus_per_task=4,
            nodes=config.training_config.num_nodes,
            gpus=config.training_config.gpus,
            constraint=config.training_config.slurm_constraint,
            logs_path=config.training_config.slurm_logs_path,
            prince=config.training_config.prince,
        )
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)

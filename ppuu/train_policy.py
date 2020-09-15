"""Train a policy / controller"""
import os

import pytorch_lightning as pl
import torch.multiprocessing

from ppuu import lightning_modules
from ppuu import slurm
from ppuu.data import NGSIMDataModule

from ppuu.train_utils import CustomLoggerWB, ModelCheckpoint


def main(config):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    if config.training_config.debug or config.training_config.fast_dev_run:
        config.training_config.set_dataset("50")
        config.training_config.epoch_size = 10
        config.training_config.n_epochs = 10
        config.cost_config.uncertainty_n_batches = 10

    module = lightning_modules.get_module(config.model_config.model_type)
    datamodule = NGSIMDataModule(
        config.training_config.dataset,
        config.training_config.epoch_size,
        config.training_config.validation_size,
        config.training_config.batch_size,
    )

    pl.seed_everything(config.training_config.seed)

    logger = CustomLoggerWB(
        save_dir=config.training_config.output_dir,
        experiment_name=config.training_config.experiment_name,
        seed=f"seed={config.training_config.seed}",
        version=config.training_config.version,
        project="PPUU_policy",
    )
    logger.log_hyperparams(module.hparams)

    period = max(1, config.training_config.n_epochs // 5)

    trainer = pl.Trainer(
        gpus=config.training_config.gpus,
        num_nodes=config.training_config.num_nodes,
        gradient_clip_val=50.0,
        max_epochs=config.training_config.n_epochs,
        check_val_every_n_epoch=period,
        num_sanity_val_steps=0,
        fast_dev_run=config.training_config.fast_dev_run,
        distributed_backend=config.training_config.distributed_backend,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                logger.log_dir, "checkpoints", "{epoch}_{success_rate}"
            ),
            save_top_k=-1,
            save_last=True,
        ),
        logger=logger,
        resume_from_checkpoint=config.training_config.resume_from_checkpoint,
        weights_save_path=logger.log_dir,
        track_grad_norm=2,
    )

    model = module(config)
    trainer.fit(model, datamodule=datamodule)
    return model


if __name__ == "__main__":
    module = lightning_modules.get_module_from_command_line()
    config = module.Config.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor(
            config.training_config.experiment_name,
            cpus_per_task=4,
            nodes=config.training_config.num_nodes,
            gpus=config.training_config.gpus,
        )
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)

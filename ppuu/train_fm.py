"""Train a forward model"""
import os
import pytorch_lightning as pl
import torch.multiprocessing

from ppuu.lightning_modules import fm
from ppuu import slurm

from ppuu.train_utils import CustomLoggerWB
from ppuu.data import NGSIMDataModule


def main(config):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    config.training.auto_batch_size()
    module = fm.get_module(config.model.model_type)
    data_module = NGSIMDataModule(
        config.training.dataset,
        config.training.epoch_size,
        config.training.validation_size,
        config.training.batch_size,
        shift=config.training.data_shift,
        random_actions=config.training.random_actions,
        npred=config.training.n_pred,
        ncond=config.training.n_cond,
        workers=0,
        diffs=config.training.diffs,
    )

    pl.seed_everything(config.training.seed)

    logger = CustomLoggerWB(
        save_dir=config.training.output_dir,
        experiment_name=config.training.experiment_name,
        seed=f"seed={config.training.seed}",
        version=config.training.version,
        project="PPUU_fm",
        offline=config.training.wandb_offline,
    )

    trainer = pl.Trainer(
        gradient_clip_val=5.0,
        max_epochs=config.training.n_epochs,
        check_val_every_n_epoch=config.training.validation_period,
        num_sanity_val_steps=0,
        fast_dev_run=config.training.fast_dev_run,
        checkpoint_callback=(
            pl.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    logger.log_dir, "checkpoints", "{epoch}_{success_rate}"
                ),
                save_top_k=None,
                monitor=None,
            )
            if logger.log_dir is not None
            else None
        ),
        logger=logger,
        gpus=config.training.gpus,
        num_nodes=config.training.num_nodes,
        distributed_backend=config.training.distributed_backend,
        weights_save_path=logger.first_log_dir,
        track_grad_norm=2,
        resume_from_checkpoint=config.training.resume_from_checkpoint,
        terminate_on_nan=True,
    )

    model = module(config)
    logger.log_hyperparams(model.hparams)
    if config.training.resume_from_checkpoint is not None:
        model.model.set_enable_latent(True)
    trainer.fit(model, data_module)
    return model


if __name__ == "__main__":
    module = fm.get_module_from_command_line()
    config = module.Config.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor(
            job_name=config.training.experiment_name,
            cpus_per_task=4,
            nodes=config.training.num_nodes,
            gpus=config.training.gpus,
            logs_path=config.training.slurm_logs_path,
            prince=config.training.prince,
        )
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)

"""Train a forward model"""
import os
import pytorch_lightning as pl
import torch.multiprocessing

from ppuu import lightning_modules
from ppuu import slurm

from train_utils import CustomLoggerWB
from ppuu.data import NGSIMDataModule


def main(config):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    module = lightning_modules.get_module(config.model.model_type)
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
    )

    pl.seed_everything(config.training.seed)

    logger = CustomLoggerWB(
        save_dir=config.training.output_dir,
        experiment_name=config.training.experiment_name,
        seed=f"seed={config.training.seed}",
        version=config.training.version,
        project="PPUU_fm",
    )

    logger.log_hyperparams(module.hparams)

    # period = max(1, config.training.n_epochs // 5)
    period = min(10, config.training.n_epochs // 10)


    trainer = pl.Trainer(
        gradient_clip_val=5.0,
        max_epochs=config.training.n_epochs,
        check_val_every_n_epoch=period,
        num_sanity_val_steps=0,
        fast_dev_run=config.training.fast_dev_run,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                logger.log_dir, "checkpoints", "{epoch}_{success_rate}"
            ),
            save_top_k=-1,
            save_last=True,
        ),
        logger=logger,
        gpus=config.training.gpus,
        num_nodes=config.training.num_nodes,
        distributed_backend=config.training.distributed_backend,
        weights_save_path=logger.first_log_dir,
        track_grad_norm=2,
    )

    model = module(config)
    if config.training.resume_from_checkpoint is not None:
        model.model.set_enable_latent(True)
    trainer.fit(model, data_module)
    return model


if __name__ == "__main__":
    module = lightning_modules.get_module_from_command_line()
    config = module.Config.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor(
            config.training.experiment_name,
            cpus_per_task=4,
            nodes=config.training.num_nodes,
            gpus=config.training.gpus,
        )
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)

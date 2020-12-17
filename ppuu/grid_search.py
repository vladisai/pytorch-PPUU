import os
import pkg_resources

import pytorch_lightning as pl
from pytorch_lightning import Callback

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from ppuu.train_policy import CustomLogger

if pkg_resources.parse_version(pl.__version__) < pkg_resources.parse_version(
    "0.7.1"
):
    raise RuntimeError(
        "PyTorch Lightning>=0.7.1 is required for this example."
    )


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
    config = Module.Config()

    config.output_dir = None

    module = Module(config)
    # Filenames for each trial must be made unique in order to access each
    # checkpoint.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join("TODO", "trial_{}".format(trial.number), "{epoch}"),
        monitor="val_acc",
    )

    # The default logger in PyTorch Lightning writes to event files to be
    # consumed by TensorBoard. We don't use any logger here as it requires us
    # to implement several abstract methods. Instead we setup a simple
    # callback, that saves metrics from each validation step.
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=21,
        gpus=1,
        callbacks=[metrics_callback],
        early_stop_callback=PyTorchLightningPruningCallback(
            trial, monitor="accuracy"
        ),
    )

    logger = CustomLogger(
        save_dir=config.training_config.output_dir,
        name=config.training_config.experiment_name,
        version=f"seed={config.training_config.seed}",
    )

    period = max(1, config.training_config.n_epochs // 5)
    trainer = pl.Trainer(
        gpus=1,
        gradient_clip_val=50.0,
        max_epochs=config.training_config.n_epochs,
        check_val_every_n_epoch=period,
        num_sanity_val_steps=0,
        checkpoint_callback=pl.callbacks.ModelCheckpoint(
            filepath=os.path.join(logger.log_dir, "checkpoints"),
            save_top_k=-1,
            save_last=True,
            run_eval=config.training_config.run_eval,
        ),
        logger=logger,
    )

    trainer.fit(module)

    return metrics_callback.metrics[-1]["val_acc"].item()


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=1, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

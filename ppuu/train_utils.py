import json
import os
from collections import defaultdict

import pytorch_lightning as pl

from ppuu import slurm
from ppuu import eval_policy


def empty_list(_):
    return []


class CustomLoggerWB(pl.loggers.WandbLogger):
    def __init__(
        self,
        *args,
        experiment_name,
        seed,
        save_dir,
        json_filename="logs.json",
        version=None,
        **kwargs,
    ):
        self.json_filename = json_filename

        self.logs = []
        self.custom_logs = defaultdict(empty_list)
        self.log_dir = os.path.join(save_dir, experiment_name, seed)

        name = experiment_name

        # find first free directory
        self.first_log_dir = self.log_dir
        if version is not None:
            name = f"{name}_{version}"
            version = f"{name}_{version}_{os.environ.get('SLURM_NODEID')}_{os.environ.get('SLURM_LOCALID')}"
            self.log_dir = f"{self.log_dir}_{version}"
        else:
            if os.path.exists(self.log_dir):
                k = 1
                while os.path.exists(f"{self.log_dir}_{k}"):
                    k += 1
                self.log_dir = f"{self.log_dir}_{k}"
                name = f"{name}_{k}"

        super().__init__(
            *args, name=name, save_dir=save_dir, version=version, **kwargs
        )

    @pl.loggers.base.rank_zero_only
    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)
        self.logs.append(metrics)

    @pl.loggers.base.rank_zero_only
    def log_custom(self, key, value):
        self.custom_logs[key].append(value)

    @pl.loggers.base.rank_zero_only
    def save(self):
        super().save()
        os.makedirs(self.log_dir, exist_ok=True)
        logs_json_save_path = os.path.join(self.log_dir, self.json_filename)
        dict_to_save = dict(custom=self.custom_logs, logs=self.logs)
        with open(logs_json_save_path, "w") as f:
            json.dump(dict_to_save, f, indent=4)


class CustomLogger(pl.loggers.TensorBoardLogger):
    def __init__(self, *args, json_filename="logs.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = []
        self.custom_logs = defaultdict(empty_list)
        self.json_filename = json_filename
        self.ctr = 0

    @pl.loggers.base.rank_zero_only
    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step)
        self.logs.append(metrics)

    @pl.loggers.base.rank_zero_only
    def log_custom(self, key, value):
        self.custom_logs[key].append(value)
        if type(value) == tuple:
            self.experiment.add_scalar(
                f"custom/{key}/{value[1]}", value[0], self.ctr
            )
        else:
            self.experiment.add_scalar(f"custom/{key}", value, self.ctr)
        self.ctr += 1

    @pl.loggers.base.rank_zero_only
    def save(self):
        super().save()
        os.makedirs(self.log_dir, exist_ok=True)
        logs_json_save_path = os.path.join(self.log_dir, self.json_filename)
        dict_to_save = dict(custom=self.custom_logs, logs=self.logs)
        with open(logs_json_save_path, "w") as f:
            json.dump(dict_to_save, f, indent=4)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, run_eval=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_eval = run_eval
        if self.run_eval:
            self.executor = slurm.get_executor(
                "eval_policy", cpus_per_task=8, cluster="slurm"
            )
            # 1 hour is enough for eval
            self.executor.update_parameters(slurm_time="1:00:00")

    def _save_model(self, filepath):
        super()._save_model(filepath)
        if self.run_eval:
            print("evaluating", filepath)
            self.eval_config = eval_policy.EvalConfig(
                checkpoint_path=filepath, save_gradients=True
            )
            self.executor.submit(eval_policy.main, self.eval_config)

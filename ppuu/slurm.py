import argparse
from dataclasses import dataclass
import yaml

import submitit

from ppuu import configs

LOG_CONFIG_PATH = "./slurm_config.yaml"


@dataclass
class SlurmConfig(configs.ConfigBase):
    logs_path: str = ""
    results_path: str = ""


def parse_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slurm", action="store_true", help="make this run on slurm",
    )
    args, _ = parser.parse_known_args()
    return args.slurm


def get_executor(
    job_name,
    cpus_per_task=1,
    cluster=None,
    nodes=1,
    gpus=1,
    constraint="turing",
    logs_path=None,
    prince=False,
):
    if logs_path is None:
        with open(LOG_CONFIG_PATH, "r") as f:
            d = yaml.safe_load(f)
            config = SlurmConfig.parse_from_dict(d)
        logs_path = config.logs_path

    executor = submitit.AutoExecutor(folder=logs_path, cluster=cluster)
    if not prince:
        executor.update_parameters(
            name=job_name,
            # slurm_time="00:10:00", # two days
            slurm_time="48:00:00",  # two days
            gpus_per_node=gpus,
            nodes=nodes,
            slurm_constraint=constraint,
            slurm_exclude="loopy8",
            cpus_per_task=cpus_per_task,
            mem_gb=100,
            slurm_ntasks_per_node=gpus,
            timeout_min=120,
        )
    else:
        executor.update_parameters(
            name=job_name,
            # slurm_time="00:10:00", # two days
            slurm_time="48:00:00",  # two days
            gpus_per_node=gpus,
            nodes=nodes,
            cpus_per_task=cpus_per_task,
            mem_gb=100,
            slurm_ntasks_per_node=gpus,
            timeout_min=120,
        )
    return executor

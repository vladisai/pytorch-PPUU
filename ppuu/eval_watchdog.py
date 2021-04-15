import glob
import os
import time
from dataclasses import dataclass

import submitit
from omegaconf import MISSING

from ppuu import configs, eval_policy, slurm


@dataclass
class EvalWatchdogConfig(configs.ConfigBase):
    dataset: str = MISSING
    contains_filter: str = "[]"
    debug: bool = False
    new_only: bool = False
    cluster: str = "slurm"
    dir: str = "."
    check_interval: int = -1


def should_run(checkpoint, contains_filter):
    res = True

    # if contains filter is not empty, match filenames
    if len(contains_filter) > 0:
        res = False
        for i in contains_filter:
            if i in checkpoint:
                res = True
    # check if evaluation result file is already there
    results_path = os.path.join(
        checkpoint.replace("checkpoints", "evaluation_results"),
        "evaluation_results.json",
    )
    alt_results_path = os.path.join(
        checkpoint.replace("checkpoints", "evaluation_results"),
        "evaluation_results_symbolic.json",
    )
    if os.path.exists(results_path):
        res = False
    if os.path.exists(alt_results_path):
        res = False
    if checkpoint.endswith("=0.ckpt"):
        res = False
    if checkpoint.endswith("last.ckpt"):
        res = False
    return res


def submit(executor, path, config):
    print("submitting", path)
    config = eval_policy.EvalConfig(
        checkpoint_path=path,
        num_processes=8,
        dataset=config.dataset,
    )
    return executor.submit(eval_policy.main, config)


def get_all_checkpoints(path, contains_filter):
    path_regex = os.path.join(path, "**/*.ckpt")
    checkpoints = glob.glob(path_regex, recursive=True)
    checkpoints = filter(lambda x: os.path.isfile(x), checkpoints)
    checkpoints = filter(lambda x: should_run(x, contains_filter), checkpoints)
    return checkpoints


def main():
    config = EvalWatchdogConfig.parse_from_command_line()

    contains_filter = config.contains_filter.split(",")
    print(contains_filter)

    executor = slurm.get_executor(
        job_name="eval", cpus_per_task=8, cluster=config.cluster
    )
    executor.update_parameters(slurm_time="2:00:00")

    already_run = []

    first_run = True
    while True:
        checkpoints = get_all_checkpoints(config.dir, contains_filter)
        for checkpoint in checkpoints:
            if checkpoint not in already_run:
                already_run.append(checkpoint)
                if config.debug:
                    print("would run", checkpoint)
                if not config.debug and (not first_run or not config.new_only):
                    job = submit(executor, checkpoint, config)
                    if job is not None:
                        if config.cluster in ["local", "debug"]:
                            print(job)
                            while True:
                                try:
                                    print(job)
                                    print(job.result())
                                    break
                                except submitit.core.utils.UncompletedJobError as e:
                                    print("waiting", str(e))
                        print("job id: ", job.job_id)
        print("done")
        if config.check_interval == -1:
            break
        time.sleep(config.check_interval)
        first_run = False


if __name__ == "__main__":
    main()

import os
import argparse
import time
import glob

import submitit

from ppuu import slurm
from ppuu import eval_policy


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
    if os.path.exists(results_path):
        res = False
    if checkpoint.endswith("=0.ckpt"):
        res = False
    if checkpoint.endswith("last.ckpt"):
        res = False
    return res


def submit(executor, path, model_type):
    print("submitting", path)
    config = eval_policy.EvalConfig(
        checkpoint_path=path,
        save_gradients=True,
        num_processes=8,
        model_type=model_type,
    )
    return executor.submit(eval_policy.main, config)


def get_all_checkpoints(path, contains_filter):
    path_regex = os.path.join(path, "**/*.ckpt")
    checkpoints = glob.glob(path_regex, recursive=True)
    checkpoints = filter(lambda x: os.path.isfile(x), checkpoints)
    checkpoints = filter(lambda x: should_run(x, contains_filter), checkpoints)
    return checkpoints


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--dir", type=str, default=".")
    parser.add_argument(
        "--check_interval",
        type=int,
        default=300,
        help="interval in seconds between checks for new results",
    )
    parser.add_argument(
        "--new_only",
        action="store_true",
        help="don't evaluate existing checkpoints",
    )
    parser.add_argument(
        "--debug", action="store_true", help="don't run jobs",
    )
    parser.add_argument(
        "--contains_filter",
        type=str,
        default="[]",
        help="run only experiments containing words from the filter list",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="continuous_v3",
        help="Policy model type",
    )
    parser.add_argument("--cluster", type=str, default="slurm")
    opt = parser.parse_args()
    contains_filter = eval(opt.contains_filter)

    executor = slurm.get_executor(
        job_name="eval", cpus_per_task=8, cluster=opt.cluster
    )
    executor.update_parameters(slurm_time="2:00:00")

    already_run = []

    first_run = True
    while True:
        checkpoints = get_all_checkpoints(opt.dir, contains_filter)
        for checkpoint in checkpoints:
            if checkpoint not in already_run:
                already_run.append(checkpoint)
                if opt.debug:
                    print("would run", checkpoint)
                if not opt.debug and (not first_run or not opt.new_only):
                    job = submit(executor, checkpoint, opt.model_type)
                    if job is not None:
                        if opt.cluster in ["local", "debug"]:
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
        if opt.check_interval == -1:
            break
        time.sleep(opt.check_interval)
        first_run = False


if __name__ == "__main__":
    main()

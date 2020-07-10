import os
import argparse
import time
import glob

from ppuu import slurm
from ppuu import eval_policy


def should_run(checkpoint, contains_filter):
    if len(contains_filter) == 0:
        return True
    res = False
    for i in contains_filter:
        if i in checkpoint:
            res = True
    return res


def submit(executor, path):
    print("submitting", path)
    if path.endswith("=0.ckpt"):
        return None
    config = eval_policy.EvalConfig(
        checkpoint_path=path, save_gradients=True, num_processes=10
    )
    if not os.path.exists(config.output_dir):
        return executor.submit(eval_policy.main, config)
    else:
        return None


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
        "--contains_filter",
        type=str,
        default='[]',
        help="run only experiments containing words from the filter list",
    )
    parser.add_argument("--cluster", type=str, default="slurm")
    opt = parser.parse_args()
    contains_filter = eval(opt.contains_filter)

    executor = slurm.get_executor(
        job_name="eval", cpus_per_task=8, cluster=opt.cluster
    )
    executor.update_parameters(slurm_time="2:00:00")

    path_regex = os.path.join(opt.dir, "**/*.ckpt")
    print(path_regex)

    already_run = []

    first_run = True
    while True:
        checkpoints = glob.glob(path_regex, recursive=True)
        for checkpoint in checkpoints:
            if not should_run(checkpoint, contains_filter):
                continue
            if checkpoint not in already_run:
                already_run.append(checkpoint)
                if not first_run or not opt.new_only:
                    job = submit(executor, checkpoint)
                    if job is not None and opt.cluster in ["local", "debug"]:
                        print(job.result())
        print("done")
        if opt.check_interval == -1:
            break
        time.sleep(opt.check_interval)
        first_run = False


if __name__ == "__main__":
    main()

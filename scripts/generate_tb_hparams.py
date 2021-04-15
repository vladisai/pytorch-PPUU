import argparse
import json
import os

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


def get_success_rate(path):
    json_path = os.path.join(path, "logs.json")
    with open(json_path, "r") as f:
        logs = json.load(f)
    for log in logs["logs"]:
        if "success_rate" in log:
            return log["success_rate"]
    if "custom" in logs:
        if "success_rate" in logs["custom"]:
            return logs["custom"]["success_rate"][0]
    return None


def get_hparams(path):
    yaml_path = os.path.join(path, "hparams.yaml")
    with open(yaml_path, "r") as f:
        hparams = yaml.load(f, Loader=yaml.Loader)
    return hparams


def get_results(path):
    success_rate = get_success_rate(path)

    ckpt_path = os.path.join(path, "last.ckpt")
    if not os.path.exists(ckpt_path):
        return None, None
    x = torch.load(ckpt_path)

    # hparams = get_hparams(path)
    hparams = x["hyper_parameters"]

    hparams_to_report = hparams["cost_config"]
    hparams_to_report["seed"] = hparams["training_config"]["seed"]
    hparams_to_report["lr"] = hparams["training_config"]["learning_rate"]
    for key in hparams_to_report:
        try:
            hparams_to_report[key] = hparams_to_report[key].item()
        except AttributeError:
            pass
    return hparams_to_report, success_rate


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        required=True,
        help="Path to the directory with the run results",
    )
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=os.path.join(args.path, "hparam"))

    for root, dirs, files in os.walk(args.path):
        if "logs.json" in files and "hparams.yaml" in files:
            hparams, success_rate = get_results(root)
            if success_rate is not None:
                writer.add_hparams(
                    hparams,
                    {"hparam/success_rate": success_rate},
                )
                print(f"wrote {root}")


if __name__ == "__main__":
    main()

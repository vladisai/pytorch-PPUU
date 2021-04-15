import os

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from dataclasses import dataclass
from typing import Optional

import torch.multiprocessing
from omegaconf import MISSING

from ppuu import configs, slurm
from ppuu.data import dataloader
from ppuu.eval import PolicyEvaluator
from ppuu.lightning_modules.policy import get_module


def get_optimal_pool_size():
    available_processes = len(os.sched_getaffinity(0))
    # we can't use more than 10, as in that case we don't fit into Gpu.
    optimal_pool_size = min(10, available_processes)
    return optimal_pool_size


@dataclass
class EvalConfig(configs.ConfigBase):
    checkpoint_path: Optional[str] = None
    alternative_checkpoint_path: Optional[str] = None
    dataset: str = MISSING
    save_gradients: bool = False
    debug: bool = False
    num_processes: int = -1
    output_dir: Optional[str] = None
    test_size_cap: Optional[int] = None
    slurm: bool = False
    model_type: Optional[str] = None
    diffs: bool = False

    def __post_init__(self):
        if self.num_processes == -1:
            self.num_processes = get_optimal_pool_size()
            logging.info(
                f"Number of processes wasn't speicifed, "
                f"going to use {self.num_processes}"
            )

        if self.output_dir is None:
            self.checkpoint_path = os.path.normpath(self.checkpoint_path)
            components = self.checkpoint_path.split(os.path.sep)
            components[-2] = "evaluation_results"
            self.output_dir = os.path.join(*components)
            if self.checkpoint_path[0] == os.path.sep:
                self.output_dir = os.path.sep + self.output_dir
            logging.info(
                f"Output dir wasn't specified, "
                f"going to save to {self.output_dir}"
            )


def main(config):
    if config.num_processes > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn")

    checkpoint = torch.load(config.checkpoint_path)

    if config.model_type is None:
        if "model_config" in checkpoint["hyper_parameters"]:
            config.model_type = checkpoint["hyper_parameters"]["model_config"][
                "model_type"
            ]
        else:
            config.model_type = checkpoint["hyper_parameters"]["model"][
                "model_type"
            ]

    test_dataset = dataloader.EvaluationDataset(
        config.dataset, "test", config.test_size_cap
    )

    Module = get_module(config.model_type)

    mpur_module = Module(checkpoint["hyper_parameters"])
    mpur_module.cuda()
    mpur_module._setup_mixout()
    mpur_module.load_state_dict(checkpoint["state_dict"], strict=False)
    mpur_module.policy_model.diffs = checkpoint["hyper_parameters"][
        "training"
    ]["diffs"]
    mpur_module._setup_normalizer(test_dataset.stats)

    alternative_module = None
    if config.alternative_checkpoint_path:
        alternative_module = Module.load_from_checkpoint(
            checkpoint_path=config.alternative_checkpoint_path
        )
        alternative_module.policy_model.diffs = config.diffs

    evaluator = PolicyEvaluator(
        test_dataset,
        config.num_processes,
        build_gradients=config.save_gradients,
        enable_logging=True,
    )
    result = evaluator.evaluate(
        mpur_module,
        output_dir=config.output_dir,
        alternative_module=alternative_module,
    )
    print(result["stats"])
    return result


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    config = EvalConfig.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor("eval", 8)
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)

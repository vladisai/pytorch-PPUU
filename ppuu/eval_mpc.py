import os

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from pathlib import Path
import yaml
from dataclasses import dataclass
import dataclasses
from typing import Optional, Any

import torch.multiprocessing

from ppuu import configs
from ppuu.data import dataloader
from ppuu.lightning_modules.policy import get_module
from ppuu.lightning_modules.fm import FM
from ppuu.eval import PolicyEvaluator
from ppuu.costs.policy_costs_km import PolicyCostKMTaper
from ppuu import slurm
from ppuu.modeling.policy_models import MPCKMPolicy
from ppuu.eval_mpc_visualizer import EvalVisualizer

from omegaconf import MISSING


def get_optimal_pool_size():
    available_processes = len(os.sched_getaffinity(0))
    # we can't use more than 10, as in that case we don't fit into Gpu.
    optimal_pool_size = min(10, available_processes)
    return optimal_pool_size


@dataclass
class EvalMPCConfig(configs.ConfigBase):
    dataset: str = MISSING
    save_gradients: bool = False
    debug: bool = False
    num_processes: int = -1
    output_dir: Optional[str] = None
    test_size_cap: Optional[int] = None
    slurm: bool = False
    cost_type: str = "km_taper"
    diffs: bool = False
    cost: PolicyCostKMTaper.Config = PolicyCostKMTaper.Config()
    mpc: MPCKMPolicy.Config = MPCKMPolicy.Config()
    visualizer: Optional[Any] = None
    forward_model_path: Optional[str] = None
    seed: int = 42


def main(config):
    if config.num_processes > 0:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    torch.manual_seed(config.seed)

    test_dataset = dataloader.EvaluationDataset(
        config.dataset, "test", config.test_size_cap
    )

    if config.forward_model_path is not None:
        m_config = FM.Config()
        m_config.model.fm_type = "km_no_action"
        m_config.model.checkpoint = config.forward_model_path
        m_config.training.enable_latent = True
        m_config.training.diffs = config.diffs
        forward_model = FM(m_config).cuda()
        forward_model._setup_normalizer(test_dataset.stats)
    else:
        forward_model = None

    if config.output_dir is not None:
        print("config", config)
        print("dict config", dataclasses.asdict(config))
        print(type(config))
        path = Path(config.output_dir)
        path.mkdir(exist_ok=True, parents=True)
        path = path / "hparams.yml"
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)

    if config.visualizer == "dump":
        config.visualizer = EvalVisualizer(config.output_dir)

    normalizer = dataloader.Normalizer(test_dataset.stats)
    cost = PolicyCostKMTaper(config.cost, None, normalizer)
    policy = MPCKMPolicy(
        forward_model, cost, normalizer, config.mpc, config.visualizer
    )
    # return policy

    evaluator = PolicyEvaluator(
        test_dataset,
        config.num_processes,
        build_gradients=config.save_gradients,
        enable_logging=True,
        visualizer=config.visualizer,
    )
    result = evaluator.evaluate(policy, output_dir=config.output_dir)
    print(result["stats"])

    return result


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    config = EvalMPCConfig.parse_from_command_line()
    use_slurm = slurm.parse_from_command_line()
    if use_slurm:
        executor = slurm.get_executor("mpc", 8, logs_path=config.output_dir)
        job = executor.submit(main, config)
        print(f"submitted to slurm with job id: {job.job_id}")
    else:
        main(config)

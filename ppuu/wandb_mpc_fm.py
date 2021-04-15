import dataclasses

import torch
import wandb
import yaml

from ppuu import configs, eval_mpc_fm

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    # Set up your default hyperparameters before wandb.init
    # so they get properly set in the sweep
    hyperparameter_defaults = {}

    # Pass your defaults to wandb.init
    run = wandb.init(
        project="sweep_mpc", config=hyperparameter_defaults, reinit=True
    )
    c_dict = dict(wandb.config)

    # translate some params from log scale to normal scale
    log_params = [
        "cost.lambda_p",
        "cost.lambda_l",
        "cost.lambda_o",
        "cost.lambda_d",
        "cost.lambda_r",
        "mpc.lambda_j_mpc",
        "cost.mask_coeff",
        "mpc.lr",
    ]
    for k in c_dict:
        if k in log_params:
            if c_dict[k] == -100:
                c_dict[k] = 0.0
            else:
                c_dict[k] = 10.0 ** c_dict[k]

    # c_dict["lr"] = c_dict["iter_reach_value"] / c_dict["n_iter"]
    # unfold_len is how many seconds into the future we want to see
    if "mpc.unfold_len" in c_dict and "mpc.timestep" in c_dict:
        c_dict["mpc.unfold_len"] = int(
            c_dict["mpc.unfold_len"] / c_dict["mpc.timestep"]
        )

    print(c_dict)

    config = configs.combine_cli_dict(eval_mpc_fm.EvalMPCConfig, c_dict)

    config.test_size_cap = 25
    config.num_processes = 4
    config.diffs = False
    config.dataset_partition = "train"

    config.cost.lambda_a = 0.0
    config.cost.u_reg = 0.0
    config.cost.lambda_j = 0.0
    config.cost.rotate = 1
    config.cost.skip_contours = True
    config.mpc.save_opt_stats = True

    # Debug
    # config.mpc.n_iter = 10
    # config.mpc.unfold_len = 10

    print("parsed config is", yaml.dump(dataclasses.asdict(config)))

    results = eval_mpc_fm.main(config)

    metrics = {
        "success_rate": results["stats"]["success_rate"],
        "mean_time": results["stats"]["mean_time"],
        "mean_distance": results["stats"]["mean_distance"],
        "mean_proximity_cost": results["stats"]["mean_proximity_cost"],
        "mean_pixel_proximity_cost": results["stats"][
            "mean_pixel_proximity_cost"
        ],
        "mean_lane_cost": results["stats"]["mean_lane_cost"],
    }

    wandb.log(metrics)

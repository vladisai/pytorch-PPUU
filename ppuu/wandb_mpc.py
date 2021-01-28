import dataclasses
import yaml

import torch
import wandb

from ppuu import eval_mpc

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
        "lambda_p",
        "lambda_l",
        "lambda_o",
        "lambda_j_mpc",
        "mask_coeff",
        "lr",
    ]
    for k in c_dict:
        if k in log_params:
            if c_dict[k] == -100:
                c_dict[k] = 0.0
            else:
                c_dict[k] = 10.0 ** c_dict[k]

    if "powers" in c_dict:
        c_dict["masks_power_x"] = c_dict["powers"]
        c_dict["masks_power_y"] = c_dict["powers"]
        del c_dict["powers"]

    c_dict["n_iter"] = int(c_dict["iter_reach_value"] / c_dict["lr"])
    # unfold_len is how many seconds into the future we want to see
    c_dict["unfold_len"] = int(c_dict["unfold_len"] / c_dict["timestep"])

    del c_dict["iter_reach_value"]

    print(c_dict)

    config = eval_mpc.EvalMPCConfig.parse_from_flat_dict(c_dict)
    config.test_size_cap = 7
    config.num_processes = 7
    config.diffs = False
    config.forward_model_path = "/home/us441/nvidia-collab/vlad/results/fm/km_no_action/fm_km_no_action_64/seed=42/checkpoints/last.ckpt"
    config.dataset = "/home/us441/nvidia-collab/vlad/traffic-data-5/state-action-cost/data_i80_v0/"
    config.dataset_partition = "train"
    config.output_dir = (
        f"/home/us441/nvidia-collab/vlad/results/mpc/grid/{wandb.run.id}"
    )

    config.cost.lambda_a = 0.0
    config.cost.lambda_j = 0.0
    config.cost.u_reg = 0.0
    config.cost.rotate = 1

    # Debug
    # config.mpc.n_iter = 10
    # config.mpc.unfold_len = 10

    print("parsed config is", yaml.dump(dataclasses.asdict(config)))

    results = eval_mpc.main(config)

    metrics = {
        "success_rate": results["stats"]["success_rate"],
        "mean_time": results["stats"]["mean_time"],
        "mean_distance": results["stats"]["mean_distance"],
        "mean_proximity_cost": results["stats"]["mean_proximity_cost"],
        "mean_pixel_proximity_cost": results["stats"]["mean_pixel_proximity_cost"],
        "mean_lane_cost": results["stats"]["mean_lane_cost"],
    }

    wandb.log(metrics)

import os
import logging
import time
import json
from typing import Optional

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy
import pandas as pd
import gym
import torch

from ppuu import dataloader


class PolicyEvaluator:
    def __init__(
        self,
        dataset: dataloader.EvaluationDataset,
        num_processes: int,
        build_gradients: bool = False,
        return_episode_data: bool = False,
        enable_logging: bool = True,
    ):
        self.dataset = dataset
        self.build_gradients = build_gradients
        self.return_episode_data = return_episode_data
        self.num_processes = num_processes
        self.enable_logging = enable_logging

        i80_env_id = "I-80-v1"
        if i80_env_id not in [e.id for e in gym.envs.registry.all()]:
            gym.envs.registration.register(
                id=i80_env_id,
                entry_point="ppuu.simulator.map_i80_ctrl:ControlledI80",
                kwargs=dict(
                    fps=10,
                    nb_states=20,
                    display=False,
                    delta_t=0.1,
                    store_simulator_video=False,
                    show_frame_count=False,
                ),
            )
        self.env = gym.make(i80_env_id)

    def get_performance_stats(self, results_per_episode):
        results_per_episode_df = pd.DataFrame.from_dict(
            results_per_episode, orient="index"
        )
        return dict(
            mean_distance=results_per_episode_df["distance_travelled"].mean(),
            mean_time=results_per_episode_df["time_travelled"].mean(),
            success_rate=results_per_episode_df["road_completed"].mean(),
            collision_rate=results_per_episode_df["has_collided"].mean(),
            off_screen_rate=results_per_episode_df["off_screen"].mean(),
        )

    def _process_one_episode(
        self, policy_model, policy_cost, car_info, index, output_dir
    ):
        inputs = self.env.reset(
            time_slot=car_info["time_slot"], vehicle_id=car_info["car_id"]
        )
        images, states, costs, actions = (
            [],
            [],
            [],
            [],
        )
        cntr = 0
        # inputs, cost, done, info = env.step(numpy.zeros((2,)))
        input_state_t0 = inputs["state"].contiguous()[-1]
        cost_sequence, action_sequence, state_sequence = [], [], []
        has_collided = False
        off_screen = False
        done = False

        while not done:
            input_images = inputs["context"].contiguous()
            input_states = inputs["state"].contiguous()

            a = policy_model(
                input_images.cuda(),
                input_states.cuda(),
                sample=True,
                normalize_inputs=True,
                normalize_outputs=True,
            )
            a = a.cpu().view(1, 2).numpy()

            action_sequence.append(a)
            state_sequence.append(input_states)
            cntr += 1

            inputs, cost, done, info = self.env.step(a[0])
            if info.collisions_per_frame > 0:
                has_collided = True
                done = True
            off_screen = info.off_screen
            images.append(input_images[-1])
            states.append(input_states[-1])
            costs.append([cost["pixel_proximity_cost"], cost["lane_cost"]])
            cost_sequence.append(cost)
            actions.append(
                (
                    (torch.tensor(a[0]) - self.dataset.stats["a_mean"])
                    / self.dataset.stats["a_std"]
                )
            )

        input_state_tfinal = inputs["state"][-1]
        images = torch.stack(images)
        states = torch.stack(states)
        costs = torch.tensor(costs)
        actions = torch.stack(actions)

        result = dict(
            time_travelled=len(images),
            distance_travelled=(
                input_state_tfinal[0] - input_state_t0[0]
            ).item(),
            road_completed=1 if cost["arrived_to_dst"] else 0,
            off_screen=off_screen,
            has_collided=has_collided,
        )

        images_3_channels = (images[:, :3] + images[:, 3:]).clamp(max=255)
        episode_data = dict(
            action_sequence=numpy.stack(action_sequence),
            state_sequence=numpy.stack(state_sequence),
            cost_sequence=numpy.stack(cost_sequence),
            images=images_3_channels,
            gradients=None,
            **result,
        )

        if self.build_gradients:
            episode_data["gradients"] = policy_cost.get_grad_vid(
                policy_model,
                dict(
                    input_images=images[:, :3].contiguous(),
                    input_states=states,
                    car_sizes=torch.tensor(
                        car_info["car_size"], dtype=torch.float32
                    ),
                ),
            )[0]

        if output_dir is not None:
            episode_data_dir = os.path.join(output_dir, "episode_data")
            episode_output_path = os.path.join(episode_data_dir, str(index))
            torch.save(episode_data, episode_output_path)

        result["index"] = index

        if self.return_episode_data:
            result["episode_data"] = episode_data

        return result

    def evaluate(
        self, module: torch.nn.Module, output_dir: Optional[str] = None,
    ):
        if output_dir is not None:
            os.makedirs(
                os.path.join(output_dir, "episode_data"), exist_ok=True
            )

        time_started = time.time()
        if self.num_processes > 0:
            executor = ProcessPoolExecutor(max_workers=self.num_processes)
        else:
            executor = ThreadPoolExecutor(max_workers=1)
        async_results = []

        # We create a copy of the cost module, but don't pass in the forward
        # model because we don't need it unless we calculate uncertainty.
        policy_cost = module.CostType(
            module.config.cost_config, None, self.dataset.stats,
        )
        module.policy_model.cuda()
        module.policy_model.stats = self.dataset.stats
        for j, data in enumerate(self.dataset):
            async_results.append(
                executor.submit(
                    self._process_one_episode,
                    module.policy_model,
                    policy_cost,
                    data,
                    j,
                    output_dir,
                )
            )

        results_per_episode = {}

        total_images = 0
        for j in range(len(async_results)):
            simulation_result = async_results[j].result()
            results_per_episode[j] = simulation_result
            total_images += simulation_result["time_travelled"]
            stats = self.get_performance_stats(results_per_episode)

            if self.enable_logging:
                log_string = " | ".join(
                    (
                        f"ep: {j + 1:3d}/{len(self.dataset)}",
                        f"time: {simulation_result['time_travelled']}",
                        (
                            f"distance:"
                            f" {simulation_result['distance_travelled']:.0f}"
                        ),
                        f"success: {simulation_result['road_completed']:d}",
                        f"success rate: {stats['success_rate']:.2f}",
                    )
                )
                logging.info(log_string)

        executor.shutdown()

        stats = self.get_performance_stats(results_per_episode)
        result = dict(results_per_episode=results_per_episode, stats=stats,)

        diff_time = time.time() - time_started
        eval_speed = total_images / diff_time
        result["stats"]["time"] = diff_time
        result["stats"]["steps evaluated per second"] = eval_speed

        if output_dir is not None:
            with open(
                os.path.join(output_dir, "evaluation_results.json"), "w"
            ) as f:
                json.dump(result, f, indent=4)

        return result

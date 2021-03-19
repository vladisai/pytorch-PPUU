import os
import logging
import time
import json
import copy
import math
from typing import Optional
from collections import deque, namedtuple

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import gym
import torch

from ppuu.data import dataloader


MAX_ENV_QUEUE_SIZE = 5


class DummyExecutor:
    def submit(self, f, *args, **kwargs):
        return DummyResult(f(*args, **kwargs))

    def shutdown(self):
        pass


class DummyResult:
    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


class PolicyEvaluator:
    def __init__(
        self,
        dataset: dataloader.EvaluationDataset,
        num_processes: int,
        build_gradients: bool = False,
        return_episode_data: bool = False,
        enable_logging: bool = True,
        rollback_seconds: int = 3,
        visualizer=None,
        pass_gt_future=False,
    ):
        self.dataset = dataset
        self.build_gradients = build_gradients
        self.return_episode_data = return_episode_data
        self.num_processes = num_processes
        self.enable_logging = enable_logging
        self.rollback_seconds = rollback_seconds
        self.visualizer = visualizer
        self.normalizer = dataloader.Normalizer(dataset.stats)
        self.pass_gt_future = pass_gt_future

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
                    dataset_path=dataset.data_dir,
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
            success_rate_alt=results_per_episode_df[
                "road_completed_alt"
            ].mean(),
            collision_rate=results_per_episode_df["has_collided"].mean(),
            collision_ahead_rate=results_per_episode_df[
                "has_collided_ahead"
            ].mean(),
            collision_behind_rate=results_per_episode_df[
                "has_collided_behind"
            ].mean(),
            off_screen_rate=results_per_episode_df["off_screen"].mean(),
            alternative_better=results_per_episode_df[
                "alternative_better"
            ].mean(),
            alternative_distance_diff=results_per_episode_df[
                "alternative_distance_diff"
            ].mean(),
            succeeded=int(results_per_episode_df["road_completed"].sum()),
            mean_proximity_cost=results_per_episode_df[
                "mean_proximity_cost"
            ].mean(),
            mean_pixel_proximity_cost=results_per_episode_df[
                "mean_pixel_proximity_cost"
            ].mean(),
            mean_lane_cost=results_per_episode_df["mean_lane_cost"].mean(),
        )

    def unfold(self, env, inputs, policy, car_size, t_limit=None):
        Unfolding = namedtuple(
            "Unfolding",
            [
                "images",
                "states",
                "costs",
                "actions",
                "env_copies",
                "done",
                "has_collided",
                "off_screen",
                "road_completed",
                "has_collided_ahead",
                "has_collided_behind",
                "mean_proximity_cost",
                "mean_pixel_proximity_cost",
                "mean_lane_cost",
            ],
        )
        TimeCapsule = namedtuple("TimeCapsule", ["env", "inputs", "cost"])

        images, states, costs, actions = (
            [],
            [],
            [],
            [],
        )
        has_collided = False
        has_collided_ahead = False
        has_collided_behind = False
        off_screen = False
        road_completed = False
        done = False

        env_copies = deque(maxlen=self.rollback_seconds)

        t = 0

        if hasattr(policy, "reset"):
            policy.reset()

        total_cost = {
            "proximity_cost": 0.0,
            "pixel_proximity_cost": 0.0,
            "lane_cost": 0.0,
        }

        while not done:
            input_images = inputs["context"].contiguous()
            input_states = inputs["state"].contiguous()

            if self.pass_gt_future:
                a = policy(
                    input_images.cuda(),
                    input_states.cuda(),
                    car_size=car_size,
                    normalize_inputs=True,
                    normalize_outputs=True,
                    gt_future=lambda : self._get_future_with_no_action(
                        env, t=policy.config.unfold_len
                    ),
                )
            else:
                a = policy(
                    input_images.cuda(),
                    input_states.cuda(),
                    car_size=car_size,
                    normalize_inputs=True,
                    normalize_outputs=True,
                )
            a = a.cpu().view(1, 2).numpy()

            # env_copy = copy.deepcopy(self.env)
            inputs, cost, done, info = env.step(a[0])

            for k in total_cost:
                if k in cost:
                    total_cost[k] += cost[k]

            if info.collisions_per_frame > 0:
                has_collided = True
            if info.collisions_per_frame_ahead > 0:
                has_collided_ahead = True

            if info.collisions_per_frame_behind > 0:
                has_collided_behind = True

            if cost["arrived_to_dst"]:
                road_completed = True

            done = (
                done
                or (has_collided and has_collided_ahead)
                or road_completed
                or off_screen
            )

            if self.visualizer is not None:
                self.visualizer.update(inputs["context"][-1].contiguous())
                if hasattr(policy.cost, 't_image'):
                    self.visualizer.update_t(
                        policy.cost.t_image.contiguous(), policy.cost.t_image_data
                    )
                    self.visualizer.update_c(policy.cost.overlay[0].contiguous())

            # every second, we save a copy of the environment
            if t % 10 == 0:
                # need to remove lane surfaces because they're unpickleable
                env._lane_surfaces = dict()
                env_copies.append(
                    TimeCapsule(copy.deepcopy(env), inputs, cost)
                )
            t += 1

            off_screen = info.off_screen
            images.append(input_images[-1])
            states.append(input_states[-1])
            costs.append(cost)
            actions.append(
                (
                    (torch.tensor(a[0]) - self.dataset.stats["a_mean"])
                    / self.dataset.stats["a_std"]
                )
            )

            if t_limit is not None and t >= t_limit:
                break

        for k in total_cost:
            total_cost[k] /= t

        images = torch.stack(images)
        states = torch.stack(states)
        actions = torch.stack(actions)
        return Unfolding(
            images,
            states,
            costs,
            actions,
            env_copies,
            done,
            has_collided,
            off_screen,
            road_completed,
            has_collided_ahead,
            has_collided_behind,
            total_cost["proximity_cost"],
            total_cost["pixel_proximity_cost"],
            total_cost["lane_cost"],
        )

    def _get_future_with_no_action(self, env, t):
        """ Build state and images for the future if all actions are 0"""
        Future = namedtuple("Future", ["images", "states"])
        env._lane_surfaces = dict()
        env = copy.deepcopy(env)
        images = []
        states = []
        for i in range(t):
            inputs, cost, done, info = env.step([0, 0])
            images.append(inputs["context"].contiguous()[-1])
            states.append(inputs["state"].contiguous()[-1])
            if done:
                return None # we fall back to using the forward model in this case.

        return Future(torch.stack(images), torch.stack(states))

    def _build_episode_data(self, unfolding):
        return dict(
            action_sequence=unfolding.actions,
            state_sequence=unfolding.states,
            cost_sequence=unfolding.costs,
            images=(unfolding.images[:, :3] + unfolding.images[:, 3:]).clamp(
                max=255
            ),
            gradients=None,
        )

    def _build_result(self, unfolding):
        return dict(
            time_travelled=len(unfolding.images),
            distance_travelled=(
                unfolding.states[-1][0] - unfolding.states[0][0]
            ).item(),
            road_completed=(
                unfolding.road_completed and not unfolding.has_collided
            ),
            road_completed_alt=(
                unfolding.road_completed and not unfolding.has_collided_ahead
            ),
            off_screen=(
                unfolding.off_screen
                and not (
                    (unfolding.road_completed and not unfolding.has_collided)
                    or (
                        unfolding.road_completed
                        and not unfolding.has_collided_ahead
                    )
                )
            ),
            has_collided=unfolding.has_collided,
            has_collided_ahead=unfolding.has_collided_ahead,
            has_collided_behind=unfolding.has_collided_behind,
            mean_proximity_cost=unfolding.mean_proximity_cost,
            mean_pixel_proximity_cost=unfolding.mean_pixel_proximity_cost,
            mean_lane_cost=unfolding.mean_lane_cost,
        )

    def _process_one_episode(
        self,
        policy_model,
        policy_cost,
        car_info,
        index,
        output_dir,
        alternative_policy=None,
    ):
        if self.visualizer is not None:
            self.visualizer.episode_reset()

        inputs = self.env.reset(
            time_slot=car_info["time_slot"], vehicle_id=car_info["car_id"]
        )
        unfolding = self.unfold(
            self.env, inputs, policy_model, car_info["car_size"]
        )
        alternative_unfolding = None
        if unfolding.has_collided and alternative_policy is not None:
            alternative_unfolding = self.unfold(
                unfolding.env_copies[0].env,
                unfolding.env_copies[0].inputs,
                alternative_policy,
                car_info["car_size"],
            )

        result = self._build_result(unfolding)
        episode_data = self._build_episode_data(unfolding)
        episode_data["result"] = result

        if alternative_unfolding is not None:
            result["alternative"] = self._build_result(alternative_unfolding)
            result["alternative_better"] = int(
                not unfolding.road_completed
                and alternative_unfolding.road_completed
            )
            alternative_distance = (
                alternative_unfolding.states[-1][0] - unfolding.states[0][0]
            ).item()
            result["alternative_distance_diff"] = (
                alternative_distance - result["distance_travelled"]
            )
            episode_data["alternative"] = self._build_episode_data(unfolding)
        else:
            result["alternative_better"] = math.nan
            result["alternative_distance_diff"] = math.nan

        if self.build_gradients:
            episode_data["gradients"] = policy_cost.get_grad_vid(
                policy_model,
                dict(
                    input_images=unfolding.images[:, :3].contiguous(),
                    input_states=unfolding.states,
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

        print("episode success: ", result["road_completed"])

        if self.return_episode_data:
            result["episode_data"] = episode_data

        if self.visualizer is not None:
            self.visualizer.save_video(index)

        return result

    def evaluate(
        self,
        module: torch.nn.Module,
        output_dir: Optional[str] = None,
        alternative_module: Optional[torch.nn.Module] = None,
    ):

        if output_dir is not None:
            os.makedirs(
                os.path.join(output_dir, "episode_data"), exist_ok=True
            )

        time_started = time.time()
        if self.num_processes > 0:
            executor = ProcessPoolExecutor(max_workers=self.num_processes)
        else:
            # executor = ThreadPoolExecutor(max_workers=1)
            executor = DummyExecutor()
        async_results = []

        # We create a copy of the cost module, but don't pass in the forward
        # model because we don't need it unless we calculate uncertainty.
        if self.build_gradients:
            policy_cost = module.CostType(
                module.config.cost,
                None,
                self.dataset.stats,
            )
        else:
            policy_cost = None

        if hasattr(module, "policy_model"):
            module.policy_model.cuda()
            module.policy_model.stats = self.dataset.stats
            if alternative_module is not None:
                alternative_module.policy_model.cuda()
                # alternative_module.policy_model.stats = self.dataset.stats
            policy_model = module.policy_model
        else:
            policy_model = module

        for j, data in enumerate(self.dataset):
            async_results.append(
                executor.submit(
                    self._process_one_episode,
                    policy_model,
                    policy_cost,
                    data,
                    j,
                    output_dir,
                    alternative_policy=alternative_module.policy_model
                    if alternative_module is not None
                    else None,
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
                        f"success_alt: {simulation_result['road_completed_alt']:d}",
                        f"success rate: {stats['success_rate']:.2f}",
                        f"success rate alt: {stats['success_rate_alt']:.2f}",
                    )
                )
                logging.info(log_string)

        executor.shutdown()

        stats = self.get_performance_stats(results_per_episode)
        result = dict(
            results_per_episode=results_per_episode,
            stats=stats,
        )

        diff_time = time.time() - time_started
        eval_speed = total_images / diff_time
        result["stats"]["time"] = diff_time
        result["stats"]["steps evaluated per second"] = eval_speed

        if output_dir is not None:
            with open(
                os.path.join(output_dir, "evaluation_results_symbolic.json"),
                "w",
            ) as f:
                json.dump(result, f, indent=4)

        return result

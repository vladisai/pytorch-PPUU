import glob
import logging
import math
import os
import pickle
import random
import re

import numpy
import torch

from ppuu.data.entities import DatasetSample, StateSequence


class DataStore:
    def __init__(self, dataset):
        data_dir = dataset

        self.images = []
        self.actions = []
        self.costs = []
        self.states = []
        self.ids = []
        self.ego_car_images = []
        self.data_dir = data_dir
        data_files = next(os.walk(data_dir))[1]
        data_files.sort()
        for df in data_files:
            combined_data_path = f"{data_dir}/{df}/all_data.pth"
            logging.info(f"Loading pickle {combined_data_path}")
            if os.path.isfile(combined_data_path):
                data = torch.load(combined_data_path)
                self.images += data.get("images")
                self.actions += data.get("actions")
                self.costs += data.get("costs")
                self.states += data.get("states")
                self.ids += data.get("ids")
                self.ego_car_images += data.get("ego_car")
            else:
                print(data_dir)
                images = []
                actions = []
                costs = []
                states = []
                ids = glob.glob(f"{data_dir}/{df}/car*.pkl")
                ids.sort()
                ego_car_images = []
                for f in ids:
                    print(f"[loading {f}]")
                    fd = pickle.load(open(f, "rb"))
                    Ta = fd["actions"].size(0)
                    # Tp = fd["pixel_proximity_cost"].size(0)
                    # Tl = fd["lane_cost"].size(0)
                    # if not(Ta == Tp == Tl): pdb.set_trace()
                    images.append(fd["images"])
                    actions.append(fd["actions"])
                    costs.append(
                        torch.cat(
                            (
                                fd.get("pixel_proximity_cost")[:Ta].view(
                                    -1, 1
                                ),
                                fd.get("lane_cost")[:Ta].view(-1, 1),
                            ),
                            1,
                        ),
                    )
                    states.append(fd["states"])
                    ego_car_images.append(fd["ego_car"])

                print(f"Saving {combined_data_path} to disk")
                torch.save(
                    {
                        "images": images,
                        "actions": actions,
                        "costs": costs,
                        "states": states,
                        "ids": ids,
                        "ego_car": ego_car_images,
                    },
                    combined_data_path,
                )
                self.images += images
                self.actions += actions
                self.costs += costs
                self.states += states
                self.ids += ids
                self.ego_car_images += ego_car_images

        self.n_episodes = len(self.images)
        splits_path = data_dir + "/splits.pth"
        if os.path.exists(splits_path):
            logging.info(f"Loading splits {splits_path}")
            splits = torch.load(splits_path)
            self.splits = dict(
                train=splits.get("train_indx"),
                val=splits.get("valid_indx"),
                test=splits.get("test_indx"),
            )
        else:
            print("[generating data splits]")
            rgn = numpy.random.RandomState(0)
            perm = rgn.permutation(self.n_episodes)
            n_train = int(math.floor(self.n_episodes * 0.8))
            n_valid = int(math.floor(self.n_episodes * 0.1))
            self.splits = dict(
                train=perm[0:n_train],
                valid=perm[n_train : n_train + n_valid],
                test=perm[n_train + n_valid :],
            )
            torch.save(self.splits, splits_path)

        stats_path = data_dir + "/data_stats_with_diff.pth"
        if os.path.isfile(stats_path):
            logging.info(f"Loading data stata {stats_path}")
            self.stats = torch.load(stats_path)
            self.a_mean = self.stats.get("a_mean")
            self.a_std = self.stats.get("a_std")
            self.s_mean = self.stats.get("s_mean")
            self.s_std = self.stats.get("s_std")
            self.s_diff_mean = self.stats.get("s_diff_mean")
            self.s_diff_std = self.stats.get("s_diff_std")
        else:
            print("[computing action stats]")
            all_actions = []
            for i in self.splits["train"]:
                all_actions.append(self.actions[i])
            all_actions = torch.cat(all_actions, 0)
            self.a_mean = torch.mean(all_actions, 0)
            self.a_std = torch.std(all_actions, 0)
            print("[computing state stats]")
            all_states = []
            all_state_diffs = []
            for i in self.splits["train"]:
                all_states.append(self.states[i][:, 0])
                c_diff = self.states[i][1:, 0] - self.states[i][:-1, 0]
                c_diff[:, 2:] = self.states[i][1:, 0, 2:]
                all_state_diffs.append(c_diff)
            all_states = torch.cat(all_states, 0)
            all_state_diffs = torch.cat(all_state_diffs, 0)
            self.s_mean = torch.mean(all_states, 0)
            self.s_std = torch.std(all_states, 0)
            self.s_diff_mean = torch.mean(all_state_diffs, 0)
            self.s_diff_std = torch.std(all_state_diffs, 0)
            self.stats = {
                "a_mean": self.a_mean,
                "a_std": self.a_std,
                "s_mean": self.s_mean,
                "s_std": self.s_std,
                "s_diff_mean": self.s_diff_mean,
                "s_diff_std": self.s_diff_std,
            }
            torch.save(
                self.stats,
                stats_path,
            )

        car_sizes_path = data_dir + "/car_sizes.pth"
        self.car_sizes = torch.load(car_sizes_path)

    def parse_car_path(path):
        splits = path.split("/")
        time_slot = splits[-2]
        car_id = int(re.findall(r"car(\d+).pkl", splits[-1])[0])
        data_files = {
            "trajectories-0400-0415": 0,
            "trajectories-0500-0515": 1,
            "trajectories-0515-0530": 2,
        }
        time_slot = data_files[time_slot]
        return time_slot, car_id


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_store,
        split,
        n_cond,
        n_pred,
        size,
        shift,
        random_actions,
        normalize=True,
        state_diffs=False,
    ):
        self.split = split
        self.data_store = data_store
        self.n_cond = n_cond
        self.n_pred = n_pred
        self.size = size
        self.random = random.Random()
        # self.random.seed(12345)
        self.normalize = normalize
        self.shift = shift
        self.random_actions = random_actions
        self.state_diffs = state_diffs
        self.normalizer = Normalizer(self.data_store.stats)

    def sample_episode(self):
        return self.random.choice(self.data_store.splits[self.split])

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # this is needed so that multiple workers don't get the same elements
        return self.get_one_example()

    def get_one_example(self):
        """
        Returns one training example, which includes input staes, and images,            # noqa
        actions, and target images and states, as well as car_id and car_size.           # noqa
                 n_cond                      n_pred                                      # noqa
        <---------------------><---------------------------------->                      # noqa
        .                     ..                                  .
        +---------------------+.                                  .  ^       ^
        |i|i|i|i|i|i|i|i|i|i|i|.  3 × 117 × 24                    .  |       |
        +---------------------+.                                  .  |inputs |
        +---------------------+.                                  .  |       |
        |s|s|s|s|s|s|s|s|s|s|s|.  4                               .  |       |
        +---------------------+.                                  .  v       |
        .                   +-----------------------------------+ .  ^       |
        .                2  |a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a| .  |actions|
        .                   +-----------------------------------+ .  v       |
        .                     +-----------------------------------+  ^       | tensors   # noqa
        .       3 × 117 × 24  |i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|  |       |
        .                     +-----------------------------------+  |       |
        .                     +-----------------------------------+  |       |
        .                  4  |s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|  |targets|
        .                     +-----------------------------------+  |       |
        .                     +-----------------------------------+  |       |
        .                  2  |c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|  |       |
        .                     +-----------------------------------+  v       v
        +---------------------------------------------------------+          ^
        |                           car_id                        |          | string    # noqa
        +---------------------------------------------------------+          v
        +---------------------------------------------------------+          ^
        |                          car_size                       |  2       | tensor    # noqa
        +---------------------------------------------------------+          v
        """
        T = self.n_cond + self.n_pred
        while True:
            s = self.sample_episode()
            # s = indx[0]
            # min is important since sometimes numbers do not align causing
            # issues in stack operation below
            episode_length = min(
                self.data_store.images[s].size(0),
                self.data_store.states[s].size(0),
            )
            if episode_length >= T:
                t = self.random.randint(0, episode_length - T)
                images = self.data_store.images[s][t : t + T]
                actions = self.data_store.actions[s][t : t + T]
                states = self.data_store.states[s][t : t + T, 0]
                # costs = self.data_store.costs[s][t : t + T]
                ids = self.data_store.ids[s]
                ego_cars = self.data_store.ego_car_images[s]
                splits = self.data_store.ids[s].split("/")
                time_slot = splits[-2]
                car_id = int(re.findall(r"car(\d+).pkl", splits[-1])[0])
                size = self.data_store.car_sizes[time_slot][car_id]
                car_sizes = torch.tensor([size[0], size[1]])
                break

        if self.state_diffs:
            # TODO: this is not tested.
            states = self.normalizer.states_to_diffs(states)

        if self.normalize:
            actions = self.normalizer.normalize_actions(actions)
            states = self.normalizer.normalize_states(states)
            images = self.normalizer.normalize_images(images)
            ego_cars = self.normalizer.normalize_images(ego_cars)

        t0 = self.n_cond
        t1 = T
        input_images = images[:t0].float().contiguous()
        input_states = states[:t0].float().contiguous()
        target_images = images[t0:t1].float().contiguous()
        target_states = states[t0:t1].float().contiguous()
        # target_costs = costs[t0:t1].float().contiguous()

        if not self.shift:
            t0 -= 1
            t1 -= 1

        actions = actions[t0:t1].float().contiguous()
        if self.random_actions:
            actions = torch.rand_like(actions)
        ego_cars = ego_cars.float().contiguous()
        car_sizes = car_sizes.float()

        conditional_state_seq = StateSequence(
            input_images,
            input_states,
            car_sizes,
            ego_cars,
        )
        target_state_seq = StateSequence(
            target_images,
            target_states,
            car_sizes,
            ego_cars,
        )

        return DatasetSample(
            conditional_state_seq,
            target_state_seq,
            actions,
            sample_split_id=ids,
            episode_id=s,
            timestep=t,
        )


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split="test", size_cap=None):
        data_dir = dataset
        self.data_dir = data_dir
        splits_path = os.path.join(data_dir, "splits.pth")
        if os.path.exists(splits_path):
            logging.info(f"Loading splits {splits_path}")
            splits = torch.load(splits_path)
            self.splits = dict(
                train=splits.get("train_indx"),
                val=splits.get("valid_indx"),
                test=splits.get("test_indx"),
            )

        car_sizes_path = os.path.join(data_dir, "car_sizes.pth")
        self.car_sizes = torch.load(car_sizes_path)
        self.split = split
        self.size = len(self.splits[self.split])
        if size_cap is not None:
            self.size = min(self.size, size_cap)

        stats_path = os.path.join(data_dir, "data_stats_with_diff.pth")
        if os.path.isfile(stats_path):
            logging.info(f"Loading data stata {stats_path}")
            stats = torch.load(stats_path)
            self.stats = stats
            self.a_mean = stats.get("a_mean")
            self.a_std = stats.get("a_std")
            self.s_mean = stats.get("s_mean")
            self.s_std = stats.get("s_std")
            self.s_diff_mean = stats.get("s_diff_mean")
            self.s_diff_std = stats.get("s_diff_std")

        self.ids = []
        data_files = next(os.walk(data_dir))[1]
        data_files.sort()
        for df in data_files:
            combined_data_path = f"{data_dir}/{df}/all_data.pth"
            if os.path.isfile(combined_data_path):
                data = torch.load(combined_data_path)
                self.ids += data.get("ids")

    @classmethod
    def from_data_store(cls, data_store, split="test", size_cap=None):
        self = cls.__new__(cls)
        self.data_dir = data_store.data_dir
        self.splits = data_store.splits
        self.car_sizes = data_store.car_sizes
        self.split = split
        self.size = len(self.splits[self.split])
        if size_cap is not None:
            self.size = min(self.size, size_cap)
        self.stats = data_store.stats
        self.ids = data_store.ids
        return self

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        car_info = self.get_episode_car_info(self.splits[self.split][i])
        return car_info

    def get_episode_car_info(self, episode):
        splits = self.ids[episode].split("/")
        time_slot_str = splits[-2]
        car_id = int(re.findall(r"car(\d+).pkl", splits[-1])[0])
        data_files_mapping = {
            "trajectories-0400-0415": 0,
            "trajectories-0500-0515": 1,
            "trajectories-0515-0530": 2,
        }
        time_slot = data_files_mapping[time_slot_str]
        car_size = self.car_sizes[time_slot_str][car_id]
        result = dict(
            time_slot=time_slot,
            time_slot_str=time_slot_str,
            car_id=car_id,
            car_size=car_size,
        )
        return result


class Normalizer:
    def __init__(self, stats):
        self.data_stats = stats

    @property
    def stats(self):
        return self.data_stats

    @property
    def stats_cuda(self):
        return {k: v.cuda().unsqueeze(0) for k, v in self.data_stats.items()}

    @classmethod
    def dummy(cls):
        return cls(
            dict(
                s_mean=torch.zeros(5),
                a_mean=torch.zeros(2),
                s_std=torch.ones(5),
                a_std=torch.ones(2),
            )
        )

    def states_to_diffs(self, states):
        """ First two numbers are pixels"""
        state_diffs = states[1:] - states[:-1]
        state_diffs = torch.cat([torch.zeros(1, 5), state_diffs], axis=0)
        state_diffs[..., 2:] = states[..., 2:]
        return state_diffs

    def normalize_states(self, states):
        device = states.device
        states = states - self.data_stats["s_mean"].view(1, 5).expand(
            states.size()
        ).to(device)
        states = states / (
            1e-8 + self.data_stats["s_std"].view(1, 5).expand(states.size())
        ).to(device)
        return states

    def unnormalize_states(self, states):
        """ From normalized to feet """
        device = states.device
        states = states * (
            1e-8 + self.data_stats["s_std"].view(1, 5).expand(states.size())
        ).to(device)
        states = states + self.data_stats["s_mean"].view(1, 5).expand(
            states.size()
        ).to(device)
        return states

    def normalize_actions(self, actions):
        device = actions.device
        actions = actions - self.data_stats["a_mean"].view(1, 2).expand(
            actions.size()
        ).to(device)
        actions = actions / (
            1e-8 + self.data_stats["a_std"].view(1, 2).expand(actions.size())
        ).to(device)
        return actions

    def unnormalize_actions(self, actions):
        device = actions.device
        actions = actions * (
            1e-8 + self.data_stats["a_std"].view(1, 2).expand(actions.size())
        ).to(device)
        actions = actions + self.data_stats["a_mean"].view(1, 2).expand(
            actions.size()
        ).to(device)
        return actions

    def normalize_images(self, images):
        return images.clone().float().div_(255.0)

    def unnormalize_images(self, images):
        return images.clone().mul_(255.0).uint8()


class UnitConverter:
    METRES_IN_FOOT = 0.3048
    LANE_WIDTH_METRES = 3.7
    LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
    PIXELS_IN_METRE = LANE_WIDTH_PIXELS / LANE_WIDTH_METRES
    """
    LOOKAHEAD is 36 meters
    LOOK sideways is 2 lane widths, which is 7.2 meters
    One 6.5 pixels per s is about 1 m/s
    """

    @classmethod
    def feet_to_m(cls, x):
        return x * cls.METRES_IN_FOOT

    @classmethod
    def m_to_feet(cls, x):
        return x / cls.METRES_IN_FOOT

    @classmethod
    def pixels_to_m(cls, x):
        return x / cls.PIXELS_IN_METRE

    @classmethod
    def m_to_pixels(cls, x):
        return x * cls.PIXELS_IN_METRE

    @classmethod
    def feet_to_pixels(cls, x):
        return cls.m_to_pixels(cls.feet_to_m(x))

    @classmethod
    def pixels_to_feet(cls, x):
        return cls.m_to_feet(cls.pixels_to_m(x))

    @classmethod
    def pixels_per_s_to_kmph(cls, x):
        return cls.pixels_to_m(x) / 1000 * 60 * 60


def overlay_ego_car(images, ego_car):
    ego_car_new_shape = [*images.shape]
    ego_car_new_shape[2] = 1
    input_ego_car = ego_car[:, 2][:, None, None].expand(ego_car_new_shape)
    input_images_with_ego = torch.cat((images.clone(), input_ego_car), dim=2)
    return input_images_with_ego


if __name__ == "__main__":
    ds = DataStore("i80")
    d = Dataset(ds, "train", 20, 30, 100)

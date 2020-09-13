"""Train the forward model"""
import os
from dataclasses import dataclass

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from ppuu.data import dataloader
from ppuu import configs
from ppuu.modeling import ForwardModelKM, ForwardModel, FwdCNN
from ppuu.lightning_modules import FM

FM_PATH = "/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/models/offroad/model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=1e-06-zdropout=0.5-gclip=5.0-warmstart=1-seed=1.step400000.model"

# FM_PATH = "/home/us441/nvidia-collab/vlad/models/fixed_fm/model=fwd-cnn-vae-fp-layers=3-bsize=64-ncond=20-npred=20-lrt=0.0001-nfeature=256-dropout=0.1-nz=32-beta=0.0-zdropout=0.0-gclip=5.0-warmstart=1-seed=1.step120000.model"

MFM_PATH = "/home/us441/nvidia-collab/vlad/results/refactored_debug/test_no_shift_20_long/seed=42/checkpoints/epoch=499_success_rate=0.ckpt"

MFM_PATH = "/home/us441/nvidia-collab/vlad/results/refactored_debug/test_no_shift_30_long/seed=42/checkpoints/epoch=420_success_rate=0.ckpt"

MFM_PATH = "/home/us441/nvidia-collab/vlad/results/refactored_debug/test_no_shift_30_vlong/seed=42/checkpoints/epoch=999_success_rate=0.ckpt"

MFM_PATH = "/home/us441/nvidia-collab/vlad/results/refactored_debug/test_no_shift_30_vlong_groupnorm/seed=42_2/checkpoints/epoch=999_success_rate=0.ckpt"


@dataclass
class Config(configs.ConfigBase):
    dataset: str = "full"
    model: bool = False
    mmodel: bool = False
    normalize: bool = True
    single_check: bool = False
    ignore_z: bool = False

    def __post_init__(self):
        if self.dataset in configs.DATASET_PATHS_MAPPING:
            self.dataset = configs.DATASET_PATHS_MAPPING[self.dataset]


def predict_all_states(states, actions, stats):
    last_state = states[:, -1]
    predicted_states = []
    for i in range(actions.shape[1]):
        next_action = actions[:, i]
        predicted_state = ForwardModelKM.predict_states(
            last_state, next_action, stats
        )
        last_state = predicted_state
        predicted_states.append(predicted_state.squeeze(1))
    return torch.stack(predicted_states, dim=1)


def dummy_stats(stats):
    res = {}
    for k in stats:
        if "std" in k:
            res[k] = torch.ones_like(stats[k])
        if "mean" in k:
            res[k] = torch.zeros_like(stats[k])
    return res


def main(config):
    if config.single_check:
        b = torch.load("bad_batch.t")
        stats = b["stats"] if config.normalize else dummy_stats(b["stats"])
        predicted_states = predict_all_states(
            b["input_states"], b["actions"], stats
        )
        res = F.mse_loss(predicted_states, b["target_states"], reduce=False)
        print(res, res.max())
        print("true states", b["target_states"])
        print("pred states", predicted_states)
        print("actions", b["actions"])
        return
    data_store = dataloader.DataStore(config.dataset)
    dataset = dataloader.Dataset(
        data_store,
        "test",
        20,
        30,
        size=1000,
        shift=False,
        random_actions=False,
        normalize=config.normalize,
    )
    dataset.random.seed(24)
    loader = DataLoader(dataset, batch_size=1, num_workers=0,)

    model = None
    if config.model:
        model = ForwardModel(FM_PATH)
        model = model.cuda()
        model = model.eval()

    if config.mmodel:
        model = FM.load_from_checkpoint(MFM_PATH)
        model = model.cuda()
        model = model.eval()
        model = model.model

    cos_loss = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
    cntr = 0

    max_batch = None
    max_mse = 0

    with torch.no_grad():
        i = 0
        mse = 0.0
        mses = []
        mse_norm = 0.0
        cos = 0.0
        total = 0
        with_0 = 0

        for b in loader:
            i += 1
            for k in b:
                if k in [
                    "input_states",
                    "input_images",
                    "actions",
                    "car_sizes",
                    "target_states",
                ]:
                    b[k] = b[k].cuda()

            if model is None:
                stats = (
                    b["stats"] if config.normalize else dummy_stats(b["stats"])
                )
                predicted_states = predict_all_states(
                    b["input_states"], b["actions"], stats
                )
            else:
                pred = model.unfold(b["actions"], b)
                predicted_states = pred["pred_states"]
            c_mse = F.mse_loss(predicted_states, b["target_states"])
            predicted_norm = (
                predicted_states[:, :, 2:].norm(dim=2).unsqueeze(-1)
            )
            predicted_directions = predicted_states[:, :, 2:] / predicted_norm
            target_norm = (
                b["target_states"][:, :, 2:].norm(dim=2).unsqueeze(-1)
            )
            target_directions = b["target_states"][:, :, 2:] / target_norm
            zz = False
            if not config.normalize and c_mse > 1e-2:
                total += 1
                for x in target_norm.squeeze():
                    if x < 1e-8:
                        with_0 += 1
                        zz = True
                        break
            if c_mse > max_mse and not zz:
                max_mse = c_mse
                max_batch = b.copy()
            if not config.ignore_z or not zz:
                mse += c_mse
                mses.append(c_mse)
                mse_norm += F.mse_loss(predicted_norm, target_norm)
                cos += (
                    1 - cos_loss(predicted_directions, target_directions)
                ).mean()
                cntr += 1

        print("total", mse / cntr)
        print("speed norm", mse_norm / cntr)
        print("cos", cos / cntr)
        mses = torch.stack(mses)
        print(
            f"std: {mses.std().item()}, "
            f"mean: {mses.mean().item()}, "
            f"max: {mses.max().item()}"
        )
        print("max_mse", max_mse)
        print("cntr", cntr)
        print(f"total {total} w/ {with_0}")

        torch.save(max_batch, "bad_batch.t")


if __name__ == "__main__":
    config = Config.parse_from_command_line()
    main(config)

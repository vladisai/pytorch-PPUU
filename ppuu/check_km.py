"""Train the forward model"""
import os
from dataclasses import dataclass
from typing import Optional

# These environment variables need to be set before
# import numpy to prevent numpy from spawning a lot of processes
# which end up clogging up the system.
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch  # noqa
import torch.nn.functional as F  # noqa
from torch.utils.data import DataLoader  # noqa
from tqdm import tqdm  # noqa

from ppuu import configs  # noqa
from ppuu.data import dataloader  # noqa
from ppuu.lightning_modules.fm import FM  # noqa
from ppuu.modeling.km import predict_states, predict_states_diff  # noqa
from ppuu.wrappers import ForwardModel  # noqa


@dataclass
class Config(configs.ConfigBase):
    dataset: str = "full_5"
    model: bool = False
    mmodel: bool = False
    normalize: bool = True
    single_check: bool = False
    ignore_z: bool = False
    shift: bool = False
    path: Optional[str] = None
    diff: bool = False


def predict_all_states(predictor, states, actions, normalizer):
    last_state = states[:, -1]
    predicted_states = []
    for i in range(actions.shape[1]):
        next_action = actions[:, i]
        predicted_state = predictor(last_state, next_action, normalizer)
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
    predictor = predict_states if not config.diff else predict_states_diff
    if config.single_check:
        b = torch.load("bad_batch.t")
        stats = b["stats"] if config.normalize else dummy_stats(b["stats"])
        predicted_states = predict_all_states(
            predictor, b["input_states"], b["actions"], stats
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
        size=250,
        shift=config.shift,
        random_actions=False,
        normalize=config.normalize,
        state_diffs=config.diff,
    )
    normalizer = dataloader.Normalizer(data_store.stats)
    dataset.random.seed(24)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
    )

    model = None
    if config.model:
        model = ForwardModel(None)
        model = model.cuda()
        model = model.eval()

    if config.mmodel:
        model = FM.load_from_checkpoint(None)
        model = model.cuda()
        model = model.eval()
        model = model.model

    if config.path is not None:
        m_config = FM.Config()
        m_config.model.fm_type = "km"
        m_config.model.checkpoint = config.path
        m_config.training.enable_latent = True
        m_config.training.diffs = False
        model = FM(m_config)
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
        i_mse = 0.0
        mses = []
        mse_norm = 0.0
        pos = 0.0
        cos = 0.0
        total = 0
        with_0 = 0

        for b in tqdm(loader):
            i += 1
            for k in b:
                if k in [
                    "input_states",
                    "input_images",
                    "actions",
                    "car_sizes",
                    "target_states",
                    "target_images",
                ]:
                    b[k] = b[k].cuda()

            if model is None:
                stats = (
                    b["stats"] if config.normalize else dummy_stats(b["stats"])
                )
                predicted_states = predict_all_states(
                    predictor, b["input_states"], b["actions"], normalizer
                )
                pred_images = None
            else:
                pred = model.unfold(b["actions"], b)
                predicted_states = pred["pred_states"]
                pred_images = pred["pred_images"]
            if pred_images is not None:
                i_mse += F.mse_loss(pred_images, b["target_images"])

            c_mse = F.mse_loss(predicted_states, b["target_states"])
            predicted_norm = predicted_states[:, :, 4].unsqueeze(-1)
            predicted_directions = predicted_states[:, :, 2:4]
            target_norm = b["target_states"][:, :, 4].unsqueeze(-1)
            target_directions = b["target_states"][:, :, 2:4]
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
                pos += F.mse_loss(
                    predicted_states[:, :, 0:2],
                    b["target_states"][:, :, 0:2],
                ).mean()
                cntr += 1

        print("states", mse / cntr)
        if i_mse > 0:
            print("images", i_mse / cntr)
        print("speed norm", mse_norm / cntr)
        print("cos", cos / cntr)
        print("pos", pos / cntr)
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

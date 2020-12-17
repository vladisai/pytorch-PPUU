"""Policy models"""
from dataclasses import dataclass

from torch import nn
import torch

from ppuu.modeling.common_models import Encoder
from ppuu.modeling.mixout import MixLinear

from ppuu import configs

from ppuu.modeling.km import predict_states


class MixoutDeterministicPolicy(nn.Module):
    def __init__(self, original_model, p):
        super().__init__()
        self.original_model = original_model
        fc_layers = []
        for layer in original_model.fc:
            if isinstance(layer, nn.Linear):
                fc_layers.append(MixLinear(layer.in_features, layer.out_features, bias=True, target=layer.weight, p=p,))
            else:
                fc_layers.append(layer)
        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self, state_images, states, normalize_inputs=False, normalize_outputs=False, car_size=None,
    ):
        bsize = state_images.size(0)
        device = state_images.device
        if normalize_inputs:
            state_images = state_images.clone().float().div_(255.0)
            if self.diffs:
                state_diffs = states.clone()
                state_diffs = state_diffs[1:] - state_diffs[:-1]
                state_diffs = torch.cat([torch.zeros(1, 5).to(device), state_diffs], axis=0)
                state_diffs[:, 2:] = states[:, 2:]
                states = state_diffs
                mean, std = self.stats["s_diff_mean"], self.stats["s_diff_std"]
            else:
                mean, std = self.stats["s_mean"], self.stats["s_std"]
            states -= mean.cuda().view(1, 5).expand(states.size())
            states /= std.cuda().view(1, 5).expand(states.size())
            if state_images.dim() == 4:  # if processing single vehicle
                state_images = state_images.cuda().unsqueeze(0)
                states = states.cuda().unsqueeze(0)

        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)  # from hidden_size to n_hidden
        a = self.fc(h).view(bsize, self.n_outputs)

        if normalize_outputs:
            a = a.data
            a.clamp_(-3, 3)
            a = self.original_model.normalizer.unnormalize_actions(a)
        return a


class DeterministicPolicy(nn.Module):
    def __init__(
        self, n_cond=20, n_feature=256, n_actions=2, h_height=14, h_width=3, n_hidden=256, diffs=False, turn_power=3,
    ):
        super().__init__()
        self.n_channels = 4
        self.n_cond = n_cond
        self.n_feature = n_feature
        self.n_actions = n_actions
        self.h_height = h_height
        self.h_width = h_width
        self.n_hidden = n_hidden
        self.diffs = diffs
        self.turn_power = turn_power
        self.encoder = Encoder(a_size=0, n_inputs=self.n_cond, n_channels=self.n_channels, batch_norm=False,)
        self.n_outputs = self.n_actions
        self.hsize = self.n_feature * self.h_height * self.h_width
        self.proj = nn.Linear(self.hsize, self.n_hidden)

        self.fc = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            # nn.BatchNorm1d(self.n_hidden, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            # nn.BatchNorm1d(self.n_hidden, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            # nn.BatchNorm1d(self.n_hidden, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_outputs),
        )

    def forward(self, state_images, states, normalize_inputs=False, normalize_outputs=False, car_size=None):
        if state_images.dim() == 4:  # if processing single vehicle
            state_images = state_images.cuda().unsqueeze(0)
            states = states.cuda().unsqueeze(0)

        bsize = state_images.size(0)
        device = state_images.device
        if normalize_inputs:
            state_images = state_images.clone().float().div_(255.0)
            if self.diffs:
                state_diffs = states.clone()
                state_diffs = state_diffs[:, 1:] - state_diffs[:, :-1]
                state_diffs = torch.cat([torch.zeros(bsize, 1, 5).to(device), state_diffs], axis=1)
                state_diffs[:, :, 2:] = states[:, :, 2:]
                states = state_diffs
                mean, std = self.stats["s_diff_mean"], self.stats["s_diff_std"]
            else:
                mean, std = self.stats["s_mean"], self.stats["s_std"]
            states -= mean.cuda().view(1, 5).expand(states.size())
            states /= std.cuda().view(1, 5).expand(states.size())

        h = self.encoder(state_images, states).view(bsize, self.hsize)
        h = self.proj(h)  # from hidden_size to n_hidden
        a_t = self.fc(h).view(bsize, self.n_outputs)
        a = a_t.clone()
        # Making the area around 0 smoother, we use e.g. x^3 as a smoother version of x.
        a[..., 1] = a_t[..., 1].pow(self.turn_power)

        if normalize_outputs:
            a = a.data
            a.clamp_(-3, 3)
            a *= self.stats["a_std"].view(1, 2).expand(a.size()).cuda()
            a += self.stats["a_mean"].view(1, 2).expand(a.size()).cuda()
        return a


class MPCKMPolicy(nn.Module):
    @dataclass
    class Config(configs.ConfigBase):
        n_iter: int = 1000
        lr: float = 0.01

    def __init__(
        self, forward_model, cost, normalizer, n_iter=20, lr=0.1, unfold_len=30, timestep=0.01, update_ref_period=10
    ):
        super().__init__()

        self.cost = cost
        self.cost.config.shifted_reference_frame = True
        self.cost.config.u_reg = 0.0
        self.cost.config.lambda_a = 0.001
        self.cost.config.lambda_p = 4.0
        self.cost.config.lambda_l = 1.0
        self.cost.config.lambda_o = 1.0
        self.cost.config.rotate = 1.0
        self.forward_model = forward_model
        self.normalizer = normalizer
        self.n_iter = n_iter
        self.lr = lr
        self.timestep = timestep
        self.unfold_len = unfold_len
        self.last_actions = None
        self.update_ref_period = update_ref_period
        self.reset()
        print(self.cost.config)

    def unfold_km(self, states, actions):
        """
        Autoregressively applies km state prediction to states with given actions.
            states shape : batch, state_dim
            actions shape : batch, unfold_len, action_dim
        Returns:
            predicted_states, shape = batch, unfold_len, state_dim
        """
        predictions = []
        for i in range(self.unfold_len):
            states = predict_states(states, actions[:, i], self.normalizer.data_stats, timestep=self.timestep)
            predictions.append(states)
        return torch.stack(predictions, dim=1)

    def unfold_fm(self, actions):
        """
        Autoregressively applies fm prediction to get reference images and states.
            states shape : batch, state_dim
            actions shape : batch, unfold_len, action_dim
        Returns:
            predicted_states, shape = batch, unfold_len, state_dim
        """

        history_images, history_states = self.get_history()
        inputs = {
            "input_images": history_images.cuda(),
            "input_states": history_states.cuda(),
            "stats": self.normalizer.stats_cuda,
        }

        actions_per_fm_timestep = int(0.1 / self.timestep)
        actions = actions.view(actions.shape[0], -1, actions_per_fm_timestep, 2)
        avg_actions = actions.mean(dim=2)

        unfolding = self.forward_model.model.unfold(actions_or_policy=avg_actions, batch=inputs, npred=3,)
        ref_images = unfolding["pred_images"].repeat_interleave(actions_per_fm_timestep, dim=1)
        ref_states = unfolding["pred_states"].repeat_interleave(actions_per_fm_timestep, dim=1)
        return ref_images, ref_states

    def reset(self):
        self.last_actions = None
        self.images = []
        self.states = []
        self.history_len = 0

    def get_history(self):
        if self.history_len < 20:
            padding_images = [self.images[0]] * (20 - self.history_len)
            padding_states = [self.states[0]] * (20 - self.history_len)
            return (
                torch.stack(padding_images + self.images).unsqueeze(0),
                torch.stack(padding_states + self.states).unsqueeze(0),
            )
        else:
            return torch.stack(self.images).unsqueeze(0), torch.stack(self.states).unsqueeze(0)

    def update_history(self, images, states):
        self.images.append(images[0, 0, :3])
        self.states.append(states[0])
        self.history_len += 1
        if self.history_len > 20:
            self.images = self.images[-20:]
            self.states = self.states[-20:]
            self.history_len = 20

    def __call__(self, images, states, normalize_inputs=False, normalize_outputs=False, car_size=None, init=None):
        device = states.device
        if normalize_inputs:
            states = self.normalizer.normalize_states(states.clone())
            images = self.normalizer.normalize_images(images)
            car_size = torch.tensor(car_size).unsqueeze(0)
        orig_shape = states.shape
        states = states[..., -1, :].view(-1, 5)
        images = images[..., -1, :, :, :].view(-1, 1, 4, 117, 24)
        self.update_history(images, states)
        if self.last_actions is not None:
            actions = torch.cat(
                (self.last_actions[:, 1:-1], self.last_actions[:, -2].unsqueeze(1).repeat(1, 2, 1)), dim=1
            )
            actions = torch.tensor(actions, requires_grad=True)
        else:
            actions = torch.zeros(states.shape[0], self.unfold_len, 2, device=device, requires_grad=True)

        optimizer = torch.optim.Adam((actions,), self.lr)

        # # One way to get reference states/images
        # ref_states = self.unfold_km(states, torch.zeros_like(actions))
        # ref_images = images.repeat(1, self.unfold_len, 1, 1, 1)

        # Another way is by using fm
        ref_images, ref_states = self.unfold_fm(actions)

        for i in range(self.n_iter):
            pred_states = self.unfold_km(states, actions)
            inputs = {
                "input_images": images,
                "input_states": states.unsqueeze(1),
                "car_sizes": car_size,
                "ref_states": ref_states,
                "ref_images": ref_images,
            }
            predictions = {
                "pred_states": pred_states,
                "pred_images": ref_images,
                "pred_actions": actions.unsqueeze(1),
            }

            # costs = self.cost.compute_state_costs_for_training(inputs, pred_images, pred_states, actions, car_size)
            optimizer.zero_grad()
            costs = self.cost.calculate_cost(inputs, predictions)
            costs["policy_loss"].backward()
            torch.nn.utils.clip_grad_norm_(actions, 0.5)
            optimizer.step()

            if (i + 1) % self.update_ref_period == 0:
                ref_images, ref_states = self.unfold_fm(actions)

        self.last_actions = actions

        actions = actions[:, 0]

        if normalize_outputs:
            actions = self.normalizer.unnormalize_actions(actions.data.clamp(-3, 3))
        print(actions)

        return actions.detach()

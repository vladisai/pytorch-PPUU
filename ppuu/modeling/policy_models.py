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
        if state_images.dim() == 4:  # if processing single vehicle
            state_images = state_images.cuda().unsqueeze(0)
            states = states.cuda().unsqueeze(0)

        bsize = state_images.size(0)
        if normalize_inputs:
            state_images = self.original_model.normalizer.normalize_images(state_images)
            if self.original_model.diffs:
                states = self.original_model.normalizer.states_to_diffs(states)
            states = self.original_model.normalizer.normalize_states(states)

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
            state_images = self.normalizer.normalize_images(state_images)
            if self.diffs:
                states = self.normalizer.states_to_diffs(states)
            states = self.normalizer.normalize_states(states)

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
        self,
        forward_model,
        cost,
        normalizer,
        visualizer=None,
        n_iter=30,
        lr=0.1,
        unfold_len=10,
        timestep=0.10,
        update_ref_period=100,
    ):
        super().__init__()

        self.cost = cost
        self.cost.config.shifted_reference_frame = True
        self.cost.config.u_reg = 0.0
        self.cost.config.lambda_a = 0.0
        self.cost.config.lambda_j = 0.0
        self.cost.config.lambda_p = 1.0
        self.cost.config.lambda_l = 0.1
        self.cost.config.lambda_o = 0.5
        self.cost.config.rotate = 1.0
        self.cost.config.safe_factor = 1.5
        self.cost.config.gamma = 0.99
        self.cost.config.masks_power_x = 2.0
        self.cost.config.masks_power_y = 2.0
        self.forward_model = forward_model
        self.normalizer = normalizer
        self.n_iter = n_iter
        self.lr = lr
        self.timestep = timestep
        self.unfold_len = unfold_len
        self.last_actions = None
        self.update_ref_period = update_ref_period
        self.visualizer = visualizer
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
            states = predict_states(states, actions[:, i], self.normalizer, timestep=self.timestep)
            predictions.append(states)
        return torch.stack(predictions, dim=1)

    def unfold_fm(self, images, states, actions):
        """
        Autoregressively applies fm prediction to get reference images and states.
            states shape : batch, state_dim
            actions shape : batch, unfold_len, action_dim
        Returns:
            predicted_images, shape = batch, unfold_len, images channels, images h, images w
            predicted_states, shape = batch, unfold_len, state_dim
        """

        inputs = {
            "input_images": images.cuda(),
            "input_states": states.cuda(),
        }

        actions_per_fm_timestep = int(0.1 / self.timestep)
        if actions_per_fm_timestep == 0 or self.unfold_len % actions_per_fm_timestep != 0:
            ref_states = self.unfold_km(states[..., -1, :].view(-1, 5), torch.zeros_like(actions))
            return images[:, -1].repeat(1, self.unfold_len, 1, 1, 1), ref_states
        else:
            actions = actions.view(actions.shape[0], -1, actions_per_fm_timestep, 2)
            avg_actions = actions.mean(dim=2)

            unfolding = self.forward_model.model.unfold(actions_or_policy=avg_actions, batch=inputs, npred=3,)
            ref_images = unfolding["pred_images"].repeat_interleave(actions_per_fm_timestep, dim=1)
            # TODO: this has to also account for the fact that other cars are probably moving at the same rate as us.
            ref_states = unfolding["pred_states"].repeat_interleave(actions_per_fm_timestep, dim=1)
            return ref_images, ref_states

    def reset(self):
        self.last_actions = None
        self.images = []
        self.states = []
        self.history_len = 0
        self.ctr = 0

    def update_history(self, images, states):
        self.images = images
        self.states.append(states[0])
        self.history_len += 1
        if self.history_len > 20:
            self.images = self.images[-20:]
            self.states = self.states[-20:]
            self.history_len = 20

    def __call__(
        self, images, states, normalize_inputs=False, normalize_outputs=False, car_size=None, init=None, metadata=None
    ):
        device = states.device
        # if self.ctr == 99:
        #     dump_dict = dict(images=images, states=states, car_size=car_size,)
        #     torch.save(dump_dict, "bad_example.dump")

        if normalize_inputs:
            states = self.normalizer.normalize_states(states.clone())
            images = self.normalizer.normalize_images(images)
            car_size = torch.tensor(car_size).unsqueeze(0)
        orig_shape = states.shape
        full_states = states.unsqueeze(0)
        full_images = images[:, :3].unsqueeze(0)
        states = states[..., -1, :].view(-1, 5)
        images = images[..., -1, :, :, :].view(-1, 1, 4, 117, 24)
        # if self.last_actions is not None:
        #     actions = torch.cat(
        #         (self.last_actions[:, 1:-1], self.last_actions[:, -2].unsqueeze(1).repeat(1, 2, 1)), dim=1
        #     )
        #     actions = torch.tensor(actions, requires_grad=True)
        # else:

        # Zero normalized maps to slight acceleration when normalized. We make sure we start from
        # true zero.
        actions = torch.zeros(states.shape[0], self.unfold_len, 2, device=device)
        if init is not None:
            actions[0, 0] = init
        actions = self.normalizer.normalize_actions(actions)
        actions.requires_grad = True

        optimizer = torch.optim.Adam((actions,), self.lr)
        self.ctr += 1

        # # One way to get reference states/images
        # ref_states = self.unfold_km(states, torch.zeros_like(actions))
        # ref_images = images.repeat(1, self.unfold_len, 1, 1, 1)
        self.cost.traj_landscape = False

        if self.visualizer:
            self.visualizer.episode_reset()

        for i in range(self.n_iter):
            if i % self.update_ref_period == 0:
                ref_images, ref_states = self.unfold_fm(full_images, full_states, actions)

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
            if i == self.n_iter - 1:
                self.cost.traj_landscape = True
            costs = self.cost.calculate_cost(inputs, predictions)
            if metadata is not None and "cost" not in metadata:
                metadata["cost"] = costs["policy_loss"]

            if i == self.n_iter - 1:
                self.cost.traj_landscape = False
            costs["policy_loss"].backward()

            # torch.nn.utils.clip_grad_norm_(actions, 0.5)
            optimizer.step()

            if self.visualizer:
                unnormalized_actions = self.normalizer.unnormalize_actions(actions.data)
                self.visualizer.update_values(
                    costs["policy_loss"].item(),
                    unnormalized_actions[0, 0, 0].item(),
                    unnormalized_actions[0, 0, 1].item(),
                )

        self.visualizer.update_plot()

        self.last_actions = actions

        actions = actions[:, 0]

        if normalize_outputs:
            actions = self.normalizer.unnormalize_actions(actions.data)
        print("final actions for", self.ctr, "are", actions)

        return actions.detach()

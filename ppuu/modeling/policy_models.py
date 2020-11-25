"""Policy models"""

from torch import nn
import torch

from ppuu.modeling.common_models import Encoder
from ppuu.modeling.mixout import MixLinear


class MixoutDeterministicPolicy(nn.Module):
    def __init__(self, original_model, p):
        super().__init__()
        self.original_model = original_model
        fc_layers = []
        for layer in original_model.fc:
            if isinstance(layer, nn.Linear):
                fc_layers.append(
                    MixLinear(
                        layer.in_features,
                        layer.out_features,
                        bias=True,
                        target=layer.weight,
                        p=p,
                    )
                )
            else:
                fc_layers.append(layer)
        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self,
        state_images,
        states,
        context=None,
        sample=True,
        normalize_inputs=False,
        normalize_outputs=False,
        n_samples=1,
    ):
        bsize = state_images.size(0)
        device = state_images.device
        if normalize_inputs:
            state_images = state_images.clone().float().div_(255.0)
            if self.diffs:
                state_diffs = states.clone()
                state_diffs = state_diffs[1:] - state_diffs[:-1]
                state_diffs = torch.cat(
                    [torch.zeros(1, 5).to(device), state_diffs], axis=0
                )
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
            a *= self.stats["a_std"].view(1, 2).expand(a.size()).cuda()
            a += self.stats["a_mean"].view(1, 2).expand(a.size()).cuda()
        return a


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        n_cond=20,
        n_feature=256,
        n_actions=2,
        h_height=14,
        h_width=3,
        n_hidden=256,
        diffs=False,
        turn_power=3,
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
        self.encoder = Encoder(
            a_size=0,
            n_inputs=self.n_cond,
            n_channels=self.n_channels,
            batch_norm=False,
        )
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

    def forward(
        self,
        state_images,
        states,
        context=None,
        sample=True,
        normalize_inputs=False,
        normalize_outputs=False,
        n_samples=1,
    ):
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
                state_diffs = torch.cat(
                    [torch.zeros(bsize, 1, 5).to(device), state_diffs], axis=1
                )
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

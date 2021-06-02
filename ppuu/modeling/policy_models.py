"""Policy models"""
from torch import nn

from ppuu.modeling.common_models import Encoder
from ppuu.modeling.mixout import MixLinear
from ppuu.data.entities import StateSequence


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
        normalize_inputs=False,
        normalize_outputs=False,
        car_size=None,
    ):
        if state_images.dim() == 4:  # if processing single vehicle
            state_images = state_images.cuda().unsqueeze(0)
            states = states.cuda().unsqueeze(0)

        bsize = state_images.size(0)
        if normalize_inputs:
            state_images = self.original_model.normalizer.normalize_images(
                state_images
            )
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
        conditional_state_seq: StateSequence,
        normalize_inputs=False,
        normalize_outputs=False,
    ):
        state_images = conditional_state_seq.images
        states = conditional_state_seq.states

        if state_images.dim() == 4:  # if processing single vehicle
            state_images = state_images.cuda().unsqueeze(0)
            states = states.cuda().unsqueeze(0)

        bsize = state_images.size(0)

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

from __future__ import annotations

import random
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn

from ppuu.data.entities import StateSequence
from ppuu.modeling.common_models import Decoder, Encoder, UNetwork


class FwdBase:
    """Base class for all forward models"""

    Unfolding = Any

    def forward_single_step(
        self,
        input_state_seq: StateSequence,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unfold serves to predict a sequence of next steps using
        the forward model.
        Return a tuple of image and state.
        """
        raise NotImplementedError()

    def unfold(
        self,
        input_state_seq: StateSequence,
        actions_or_policy: Union[torch.nn.Module, torch.Tensor],
        npred: int = None,
    ) -> Unfolding:
        """Unfold serves to predict a sequence of next steps using
        the forward model.
        If actions tensor is passed, we unroll for the length of the tensor.
        If policy is passed, we use npred variable to determine how far to unfold.
        """
        raise NotImplementedError()


class FwdCNN_VAE(torch.nn.Module):
    class Unfolding(NamedTuple):
        state_seq: StateSequence
        actions: torch.Tensor
        z: torch.Tensor

        def map(self, f) -> FwdCNN_VAE.Unfolding:
            return FwdCNN_VAE.Unfolding(
                self.state_seq.map(f), f(self.actions), f(self.z)
            )

    class ForwardResult(NamedTuple):
        """A tuple to hold values for fm forward result"""

        state_seq: StateSequence
        z: torch.Tensor
        p_loss: torch.Tensor

    class ForwardSingleStepResult(NamedTuple):
        """A tuple to hold values for fm forward single step result"""

        pred_image: torch.Tensor
        pred_state: torch.Tensor
        z: torch.Tensor
        kld_loss: torch.Tensor

    class SampleZResult(NamedTuple):
        """A tuple to hold sampled z and associated losses"""

        z: torch.Tensor
        kld: torch.Tensor  # KL-divergence

    def __init__(
        self,
        layers,
        nfeature,
        dropout,
        h_height,
        h_width,
        height,
        width,
        n_actions,
        hidden_size,
        ncond,
        predict_state,
        nz,
        enable_kld,
        enable_latent,
    ):
        super().__init__()
        self.layers = layers
        self.nfeature = nfeature
        self.dropout = dropout
        self.h_height = h_height
        self.h_width = h_width
        self.height = height
        self.width = width
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.ncond = ncond
        self.predict_state = predict_state
        self.normalizer = None
        self.nz = nz
        self.enable_kld = enable_kld
        self.enable_latent = enable_latent

        self.encoder = Encoder(a_size=0, n_inputs=self.ncond)
        self.decoder = Decoder(
            layers=self.layers,
            n_feature=self.nfeature,
            dropout=self.dropout,
            h_height=self.h_height,
            h_width=self.h_width,
            height=self.height,
            width=self.width,
            state_dimension=5 if self.predict_state else 0,
        )
        self.a_encoder = nn.Sequential(
            nn.Linear(self.n_actions, self.nfeature),
            # nn.BatchNorm1d(self.nfeature, momentum=0.01),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nfeature, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nfeature, self.hidden_size),
        )
        self.u_network = UNetwork(
            n_feature=self.nfeature, layers=self.layers, dropout=self.dropout
        )
        self.y_encoder = Encoder(a_size=0, n_inputs=1, states=False)

        self.z_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nfeature, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nfeature, 2 * self.nz),
        )

        self.z_expander = nn.Linear(self.nz, self.hidden_size)

    def set_enable_latent(self, value=True):
        self.enable_latent = value

    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample_z(self, *shape):
        z = torch.randn(*shape, self.nz)
        return z

    def forward_single_step(
        self,
        input_state_seq: StateSequence,
        action: torch.Tensor,
        z: Union[torch.Tensor, Callable[[torch.Tensor], SampleZResult]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Z can either be a tensor, or a function that can be called
        to obtain the value and kld_loss.
        """
        bsize = input_state_seq.images.shape[0]
        # encode the inputs (without the action)

        input_encoding = self.encoder(
            input_state_seq.images, input_state_seq.states
        )
        input_encoding = input_encoding.view(
            bsize, self.nfeature, self.h_height, self.h_width
        )

        if not self.enable_latent:
            z_exp = 0
            z_val = torch.zeros(bsize, self.nz).to(action.device)
            kld_loss = 0
        else:
            if not torch.is_tensor(z):
                z_val, kld_loss = z(input_encoding)
            else:
                z_val = z
                kld_loss = 0
            z_exp = self.z_expander(z_val).view(
                bsize, self.nfeature, self.h_height, self.h_width
            )

        h = input_encoding + z_exp
        a_emb = self.a_encoder(action).view(h.size())
        h = h + a_emb
        h = h + self.u_network(h)

        pred_image, pred_state = self.decoder(h)
        pred_image = torch.sigmoid(
            pred_image + input_state_seq.images[:, -1].unsqueeze(1)
        )
        pred_state = pred_state + input_state_seq.states[:, -1]
        return FwdCNN_VAE.ForwardSingleStepResult(
            pred_image, pred_state, z_val, kld_loss
        )

    def get_z_sampler(
        self,
        target_image: torch.Tensor,
        target_state: torch.Tensor,
        z_dropout: float = 0.0,
    ) -> Callable[torch.Tensor, SampleZResult]:
        def res(h_x: torch.Tensor) -> FwdCNN_VAE.SampleZResult:
            """Builds a lambda function that given encoding of input
            will return the latent and associated loss"""
            # we are training or estimating z distribution
            # encode the targets into z
            h_y = self.y_encoder(target_image.unsqueeze(1).contiguous())
            bsize = target_image.shape[0]
            if random.random() < z_dropout:
                z = self.sample_z(bsize, method=None, h_x=h_x).data.to(
                    target_image.device
                )
            else:
                mu_logvar = self.z_network((h_x + h_y).view(bsize, -1)).view(
                    bsize, 2, self.nz
                )
                mu = mu_logvar[:, 0]
                logvar = mu_logvar[:, 1]
                z = self.reparameterize(mu, logvar, True)
                # this can go to inf when taking exp(), so clamp it
                logvar = torch.clamp(logvar, max=4)
                if self.enable_kld:
                    kld = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    )
                    kld /= bsize
                else:
                    raise ValueError
            return FwdCNN_VAE.SampleZResult(z, kld)

        return res

    def forward(
        self,
        input_state_seq: StateSequence,
        actions: torch.Tensor,
        target_state_seq: StateSequence,
        sampling: Any = None,
        z_dropout: float = 0.0,
        z_seq: Optional[torch.Tensor] = None,
    ):
        """Main function used for forward prop. It applies the forward model
        step by step, autoregressively, while inferring or sampling the latent
        variable, depending on sampling parameter. Z latent is dropped out, and
        can also be specified by a parameter z_seq.
        We can\'t just do unfold here because of sampling/inferring z at each
        timestep.
        """
        input_images, input_states = input_state_seq[:2]
        target_images, target_states = target_state_seq[:2]
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.n_actions)
        npred = actions.size(1)
        ploss = torch.zeros(1).to(input_images.device)

        pred_images, pred_states = [], []
        z_list = []

        z = None
        for t in range(npred):
            if not self.enable_latent:
                z = torch.zeros(bsize, self.nz).to(input_images.device)
            elif sampling is None:
                z = self.get_z_sampler(
                    target_images[:, t], target_states[:, t], z_dropout
                )
            else:
                if z_seq is not None:
                    z = z_seq[t]
                else:
                    z = self.sample_z(bsize).to(input_images.device)

            pred_image, pred_state, z_val, kld_loss = self.forward_single_step(
                input_state_seq, actions[:, t], z
            )
            ploss += kld_loss
            input_state_seq = input_state_seq.shift_add(pred_image, pred_state)
            pred_images.append(pred_image)
            pred_states.append(pred_state)
            z_list.append(z_val)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_state_seq = StateSequence(
            pred_images,
            pred_states,
            input_state_seq.car_size,
            input_state_seq.ego_car_image,
        )
        z_list = torch.stack(z_list, 1)
        return FwdCNN_VAE.ForwardResult(pred_state_seq, z_list, ploss)

    def unfold(
        self,
        input_state_seq: StateSequence,
        actions_or_policy: Union[torch.nn.Module, torch.Tensor],
        npred: int = None,
        Z: Optional[torch.Tensor] = None,
    ) -> StateSequence:
        """This is almost the same as in FwdCNN, with the exception of handling
        the latent variable. If its none, its sampled, otherwise we use
        the passed one.
        """
        if torch.is_tensor(actions_or_policy):
            npred = actions_or_policy.size(1)
        else:
            # If the source of actions is a policy and not a tensor array, we
            # need a version of the state with ego car on the 4th channel.
            input_state_seq = input_state_seq.with_ego()
            assert (
                npred is not None
            ), "npred is required if unfolding for policy"

        bsize = input_state_seq.images.shape[0]
        device = input_state_seq.images.device
        if not self.enable_latent:
            Z = torch.zeros(bsize, npred, self.nz).to(device)
        else:
            if Z is None:
                Z = (
                    self.sample_z(npred * bsize)
                    .view(bsize, npred, -1)
                    .to(device)
                )

        pred_images, pred_states, pred_actions = [], [], []

        for t in range(npred):
            if torch.is_tensor(actions_or_policy):
                actions = actions_or_policy[:, t]
            else:
                actions = actions_or_policy(input_state_seq)

            pred_image, pred_state, _, _ = self.forward_single_step(
                input_state_seq, actions, Z[:, t]
            )
            input_state_seq = input_state_seq.shift_add(pred_image, pred_state)

            pred_images.append(pred_image)
            pred_states.append(pred_state)
            pred_actions.append(actions)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        pred_actions = torch.stack(pred_actions, 1)
        pred_state_seq = StateSequence(
            pred_images,
            pred_states,
            input_state_seq.car_size,
            input_state_seq.ego_car_image,
        )

        return FwdCNN_VAE.Unfolding(pred_state_seq, pred_actions, Z)

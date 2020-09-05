from typing import NamedTuple, List

import torch
import torch.nn as nn
import random

from ppuu.modeling.common_models import Encoder, UNetwork, Decoder


###############
# Main models
###############


class FMResult(NamedTuple):
    pred_images: torch.Tensor
    pred_states: torch.Tensor
    z_list: List[torch.Tensor]
    p_loss: torch.Tensor


# forward model, deterministic (compatible with TEN3 model, use to initialize)
class FwdCNN(nn.Module):
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
    ):
        super(FwdCNN, self).__init__()

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

        self.encoder = Encoder(Encoder.Config(a_size=0, n_inputs=self.ncond))
        self.decoder = Decoder(
            layers=self.layers,
            n_feature=self.nfeature,
            dropout=self.dropout,
            h_height=self.h_height,
            h_width=self.h_width,
            height=self.height,
            width=self.width,
        )
        self.a_encoder = nn.Sequential(
            nn.Linear(self.n_actions, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.BatchNorm1d(self.nfeature, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nfeature, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.BatchNorm1d(self.nfeature, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nfeature, self.hidden_size),
        )
        self.u_network = UNetwork(
            n_feature=self.nfeature, layers=self.layers, dropout=self.dropout
        )
        # else:
        #     print("[initializing encoder and decoder with: {}]".format(mfile))
        #     self.mfile = mfile
        #     pretrained_model = torch.load(mfile)["model"]
        #     self.encoder = pretrained_model.encoder
        #     self.decoder = pretrained_model.decoder
        #     self.a_encoder = pretrained_model.a_encoder
        #     self.u_network = pretrained_model.u_network
        #     self.encoder.n_inputs = self.ncond

    # dummy function
    def sample_z(self, bsize, method=None):
        return torch.zeros(bsize, 32)

    def encode(self, input_images, input_states, action):
        bsize = input_images.size(0)
        h_x = self.encoder(input_images, input_states)
        h_x = h_x.view(bsize, self.nfeature, self.h_height, self.h_width)
        a_emb = self.a_encoder(action).view(h_x.size())

        h = h_x
        h = h + a_emb
        h = h + self.u_network(h)
        return h

    def forward_single_step(self, input_images, input_states, action, z):
        h = self.encode(input_images, input_states, action)
        pred_image, pred_state = self.decoder(h)
        pred_image = torch.sigmoid(
            pred_image + input_images[:, -1].unsqueeze(1)
        )
        pred_state = pred_state + input_states[:, -1]
        return pred_image, pred_state

    def forward(self, inputs, actions, target, sampling=None, z_dropout=None):
        npred = actions.size(1)
        input_images, input_states = inputs
        pred_images, pred_states = [], []
        ploss = torch.zeros(1).to(input_images.device)
        for t in range(npred):
            pred_image, pred_state = self.forward_single_step(
                input_images, input_states, actions[:, t], None
            )
            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat(
                (input_states[:, 1:], pred_state.unsqueeze(1)), 1
            )
            pred_images.append(pred_image)
            pred_states.append(pred_state)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        return FMResult(pred_images, pred_states, None, ploss)


# this version adds the actions *after* the z variables
class FwdCNN_VAE(FwdCNN):
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
        nz,
        enable_kld,
        enable_latent,
    ):
        super(FwdCNN_VAE, self).__init__(
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
        )
        self.nz = nz
        self.enable_kld = enable_kld
        self.enable_latent = enable_latent

        #     print("[initializing encoder and decoder with: {}]".format(mfile))
        #     self.mfile = mfile
        #     pretrained_model = torch.load(mfile)
        #     if type(pretrained_model) is dict:
        #         pretrained_model = pretrained_model["model"]
        #     self.encoder = pretrained_model.encoder
        #     self.decoder = pretrained_model.decoder
        #     self.a_encoder = pretrained_model.a_encoder
        #     self.u_network = pretrained_model.u_network
        #     self.encoder.n_inputs = opt.ncond
        #     self.decoder.n_out = 1

        self.y_encoder = Encoder(
            Encoder.Config(a_size=0, n_inputs=1, states=False)
        )

        self.z_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.BatchNorm1d(self.nfeature, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.nfeature, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.BatchNorm1d(self.nfeature, momentum=0.01),
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

    def sample_z(self, bsize, method=None, h_x=None):
        z = torch.randn(bsize, self.nz)
        return z

    def forward_single_step(self, input_images, input_states, action, z):
        # encode the inputs (without the action)
        if not self.enable_latent:
            return super().forward_single_step(
                input_images, input_states, action, z
            )
        batch_size = input_images.size(0)
        z_exp = self.z_expander(z).view(
            batch_size, self.nfeature, self.h_height, self.h_width,
        )
        h = self.encode(input_images, input_states, action) + z_exp
        pred_image, pred_state = self.decoder(h)
        pred_image = torch.sigmoid(
            pred_image + input_images[:, -1].unsqueeze(1)
        )
        pred_state = pred_state + input_states[:, -1]

        return pred_image, pred_state

    def forward(
        self,
        inputs,
        actions,
        targets,
        save_z=False,
        sampling=None,
        z_dropout=0.0,
        z_seq=None,
    ):
        if not self.enable_latent:
            return super().forward(
                inputs, actions, targets, sampling, z_dropout,
            )
        input_images, input_states = inputs
        bsize = input_images.size(0)
        actions = actions.view(bsize, -1, self.n_actions)
        npred = actions.size(1)
        ploss = torch.zeros(1).to(input_images.device)

        pred_images, pred_states = [], []
        z_list = []

        z = None
        for t in range(npred):
            # encode the inputs (without the action)
            h_x = self.encoder(input_images, input_states)
            if sampling is None:
                # we are training or estimating z distribution
                target_images, target_states = targets
                # encode the targets into z
                h_y = self.y_encoder(
                    target_images[:, t].unsqueeze(1).contiguous()
                )
                if random.random() < z_dropout:
                    z = self.sample_z(bsize, method=None, h_x=h_x).data.to(
                        input_images.device
                    )
                else:
                    mu_logvar = self.z_network(
                        (h_x + h_y).view(bsize, -1)
                    ).view(bsize, 2, self.nz)
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
                        ploss += kld
                    else:
                        raise ValueError
            else:
                if z_seq is not None:
                    z = z_seq[t]
                else:
                    z = self.sample_z(bsize, method=None, h_x=h_x).to(
                        input_images.device
                    )

            z_list.append(z)
            z_exp = self.z_expander(z).view(
                bsize, self.nfeature, self.h_height, self.h_width
            )
            h_x = h_x.view(bsize, self.nfeature, self.h_height, self.h_width)
            h = h_x + z_exp
            a_emb = self.a_encoder(actions[:, t]).view(h.size())
            h = h + a_emb
            h = h + self.u_network(h)

            pred_image, pred_state = self.decoder(h)
            # if sampling is not None:
            #     pred_image.detach()
            #     pred_state.detach()
            pred_image = torch.sigmoid(
                pred_image + input_images[:, -1].unsqueeze(1)
            )
            pred_state = pred_state + input_states[:, -1]

            input_images = torch.cat((input_images[:, 1:], pred_image), 1)
            input_states = torch.cat(
                (input_states[:, 1:], pred_state.unsqueeze(1)), 1
            )
            pred_images.append(pred_image)
            pred_states.append(pred_state)

        pred_images = torch.cat(pred_images, 1)
        pred_states = torch.stack(pred_states, 1)
        z_list = torch.stack(z_list, 1)
        return FMResult(pred_images, pred_states, z_list, ploss)

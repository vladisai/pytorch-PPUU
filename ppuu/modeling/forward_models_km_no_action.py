from typing import Callable, Tuple, Union

import torch

from ppuu.data.entities import StateSequence
from ppuu.modeling.forward_models import FwdCNN_VAE
from ppuu.modeling.km import predict_states


class FwdCNNKMNoAction_VAE(FwdCNN_VAE):
    """Altered forward model that doesn't predict action,
    it uses the kinematic model prediction of next state instead."""

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
        state_predictor=predict_states,
    ):
        super().__init__(
            layers,
            nfeature,
            dropout,
            h_height,
            h_width,
            height,
            width,
            5,  # n_actions, we use km predicted state instead of the action
            hidden_size,
            ncond,
            False,  # predict_state
            nz,
            enable_kld,
            enable_latent,
        )
        self.state_predictor = state_predictor
        # We overwrite this once child models were created.
        self.n_actions = n_actions

    def forward_single_step(
        self,
        input_state_seq: StateSequence,
        action: torch.Tensor,
        z: Union[
            torch.Tensor, Callable[[torch.Tensor], FwdCNN_VAE.SampleZResult]
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """We don't use the action for action encoder, we use the true state."""

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
        pred_state = self.state_predictor(
            input_state_seq.states[:, -1], action
        )

        h = input_encoding + z_exp
        a_emb = self.a_encoder(pred_state).view(h.size())
        h = h + a_emb
        h = h + self.u_network(h)

        pred_image = self.decoder(h)
        pred_image = torch.sigmoid(
            pred_image + input_state_seq.images[:, -1].unsqueeze(1)
        )
        return FwdCNN_VAE.ForwardSingleStepResult(
            pred_image, pred_state, z_val, kld_loss
        )

from typing import Callable, Tuple, Union

import torch

from ppuu.data.entities import StateSequence
from ppuu.modeling.forward_models import FwdCNN_VAE
from ppuu.modeling.km import StatePredictor, predict_states


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

    @classmethod
    def load_from_file(cls, file_path):
        model = torch.load(file_path)
        state_dict = model["state_dict"]
        if list(state_dict.keys())[0].startswith("model."):
            for key in list(state_dict.keys()):
                state_dict[key.replace("model.", "")] = state_dict.pop(key)
        # normalizer is missing for now
        state_predictor = StatePredictor(
            diff=model["hyper_parameters"]["training"]["diffs"],
            normalizer=None,
        )
        res = cls(
            layers=model["hyper_parameters"]["model"]["layers"],
            nfeature=model["hyper_parameters"]["model"]["nfeature"],
            dropout=model["hyper_parameters"]["model"]["dropout"],
            h_height=model["hyper_parameters"]["model"]["h_height"],
            h_width=model["hyper_parameters"]["model"]["h_width"],
            height=model["hyper_parameters"]["model"]["height"],
            width=model["hyper_parameters"]["model"]["width"],
            n_actions=model["hyper_parameters"]["model"]["n_actions"],
            hidden_size=model["hyper_parameters"]["model"]["hidden_size"],
            ncond=model["hyper_parameters"]["model"]["ncond"],
            nz=model["hyper_parameters"]["model"]["nz"],
            enable_kld=True,
            enable_latent=True,
            state_predictor=state_predictor,
        )
        res.encoder.n_channels = 3
        res.load_state_dict(state_dict)
        return res

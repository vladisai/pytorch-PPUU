"""Costs calculation for policy. Calculates uncertainty and state costs.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ppuu import configs


class PolicyCostBase(ABC):
    @abstractmethod
    def calculate_cost(inputs, predictions):
        pass


class PolicyCost(PolicyCostBase):
    @dataclass
    class Config(configs.ConfigBase):
        """Configuration of cost calculation"""

        u_reg: float = 0.05
        lambda_a: float = 0.0
        lambda_j: float = 0.0
        lambda_l: float = 0.2
        lambda_o: float = 1.0
        lambda_p: float = 1.0
        gamma: float = 0.99
        u_hinge: float = 0.5
        dreaming_z_reg: float = 0.1
        uncertainty_n_pred: int = 30
        uncertainty_n_models: int = 10
        uncertainty_n_batches: int = 100
        artifact_power: int = 2
        safe_factor: float = 1.5
        skip_contours: bool = True

    def __init__(self, config, forward_model, normalizer):
        self.config = config
        self.forward_model = forward_model
        self.normalizer = normalizer
        self.traj_landscape = False

    def compute_lane_cost(self, images, car_size):
        SCALE = 0.25
        bsize, npred, nchannels, crop_h, crop_w = images.size()
        images = images.view(bsize * npred, nchannels, crop_h, crop_w)

        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * SCALE * (0.3048 * 24 / 3.7)  # pixels
        length = length * SCALE * (0.3048 * 24 / 3.7)  # pixels

        # Create separable proximity mask
        width.fill_(24 * SCALE / 2)

        max_x = torch.ceil((crop_h - length) / 2)
        #    max_y = torch.ceil((crop_w - width) / 2)
        max_y = torch.ceil(torch.zeros(width.size()).fill_(crop_w) / 2)
        max_x = (
            max_x.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        max_y = (
            max_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        min_y = torch.ceil(
            crop_w / 2 - width
        )  # assumes other._width / 2 = self._width / 2
        min_y = (
            min_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .cuda()
        )
        x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2

        x_filter = (
            x_filter.unsqueeze(0)
            .expand(bsize * npred, crop_h)
            .type(car_size.type())
            .cuda()
        )

        x_filter = torch.min(
            x_filter, max_x.view(bsize * npred, 1).expand(x_filter.size())
        )
        x_filter = (
            x_filter == max_x.unsqueeze(1).expand(x_filter.size())
        ).float()

        y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
        y_filter = (
            y_filter.view(1, crop_w)
            .expand(bsize * npred, crop_w)
            .type(car_size.type())
            .cuda()
        )
        #    y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
        y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
        y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (
            max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1)
        )
        x_filter = x_filter.cuda()
        y_filter = y_filter.cuda()
        x_filter = x_filter.type(y_filter.type())
        proximity_mask = torch.bmm(
            x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w)
        )
        proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
        images = images.view(bsize, npred, nchannels, crop_h, crop_w)
        costs = torch.max(
            (proximity_mask * images[:, :, 0].float()).view(bsize, npred, -1),
            2,
        )[0]
        return costs.view(bsize, npred), proximity_mask

    def compute_offroad_cost(self, images, proximity_mask):
        bsize, npred, nchannels, crop_h, crop_w = images.size()
        images = images.view(bsize, npred, nchannels, crop_h, crop_w)
        costs = torch.max(
            (proximity_mask * images[:, :, 2].float()).view(bsize, npred, -1),
            2,
        )[0]
        return costs.view(bsize, npred)

    def compute_proximity_cost(
        self,
        images,
        states,
        car_size=(6.4, 14.3),
        green_channel=1,
        unnormalize=False,
    ):
        SCALE = 0.25
        safe_factor = 1.5
        bsize, npred, nchannels, crop_h, crop_w = images.size()
        images = images.view(bsize * npred, nchannels, crop_h, crop_w)
        states = states.view(bsize * npred, 5).clone()

        if unnormalize:
            states = self.normalizer.unnormalize_states(states)

        speed = states[:, 4] * SCALE  # pixel/s
        width, length = car_size[:, 0], car_size[:, 1]  # feet
        width = width * SCALE * (0.3048 * 24 / 3.7)  # pixels
        length = length * SCALE * (0.3048 * 24 / 3.7)  # pixels

        safe_distance = (
            torch.abs(speed) * safe_factor + (1 * 24 / 3.7) * SCALE
        )  # plus one metre (TODO change)

        # Compute x/y minimum distance to other vehicles (pixel version)
        # Account for 1 metre overlap (low data accuracy)
        alpha = 1 * SCALE * (24 / 3.7)  # 1 m overlap collision
        # Create separable proximity mask

        max_x = torch.ceil((crop_h - torch.clamp(length - alpha, min=0)) / 2)
        max_y = torch.ceil((crop_w - torch.clamp(width - alpha, min=0)) / 2)
        max_x = (
            max_x.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(images.device)
        )
        max_y = (
            max_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(images.device)
        )

        min_x = torch.clamp(max_x - safe_distance, min=0)
        min_y = torch.ceil(
            crop_w / 2 - width
        )  # assumes other._width / 2 = self._width / 2
        min_y = (
            min_y.view(bsize, 1)
            .expand(bsize, npred)
            .contiguous()
            .view(bsize * npred)
            .to(images.device)
        )

        x_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_h))) * crop_h / 2
        x_filter = (
            x_filter.unsqueeze(0)
            .expand(bsize * npred, crop_h)
            .type(car_size.type())
            .to(images.device)
        )
        x_filter = torch.min(
            x_filter, max_x.view(bsize * npred, 1).expand(x_filter.size())
        )
        x_filter = torch.max(x_filter, min_x.view(bsize * npred, 1))

        x_filter = (x_filter - min_x.view(bsize * npred, 1)) / (
            max_x - min_x
        ).view(bsize * npred, 1)
        y_filter = (1 - torch.abs(torch.linspace(-1, 1, crop_w))) * crop_w / 2
        y_filter = (
            y_filter.view(1, crop_w)
            .expand(bsize * npred, crop_w)
            .type(car_size.type())
            .to(images.device)
        )
        y_filter = torch.min(y_filter, max_y.view(bsize * npred, 1))
        y_filter = torch.max(y_filter, min_y.view(bsize * npred, 1))
        y_filter = (y_filter - min_y.view(bsize * npred, 1)) / (
            max_y.view(bsize * npred, 1) - min_y.view(bsize * npred, 1)
        )
        x_filter = x_filter.to(images.device)
        y_filter = y_filter.to(images.device)
        proximity_mask = torch.bmm(
            x_filter.view(-1, crop_h, 1), y_filter.view(-1, 1, crop_w)
        )
        proximity_mask = proximity_mask.view(bsize, npred, crop_h, crop_w)
        images = images.view(bsize, npred, nchannels, crop_h, crop_w)
        costs = torch.max(
            (proximity_mask * images[:, :, green_channel].float()).view(
                bsize, npred, -1
            ),
            2,
        )[0]
        return dict(costs=costs, masks=proximity_mask)

    def compute_uncertainty_batch(
        self,
        batch,
        Z=None,
        estimation=True,
    ):
        """Estimates prediction uncertainty using dropout."""
        if estimation:
            torch.set_grad_enabled(False)
        input_images = batch["input_images"]
        input_states = batch["input_states"]
        actions = batch["actions"][..., : self.config.uncertainty_n_pred, :]
        car_sizes = batch["car_sizes"]

        bsize, ncond, channels, height, width = input_images.shape

        if Z is None:
            Z = self.forward_model.sample_z(
                bsize * self.config.uncertainty_n_pred, method="fp"
            )
            if type(Z) is list:
                Z = Z[0]
            Z = Z.view(bsize, self.config.uncertainty_n_pred, -1)
        Z_rep = Z.unsqueeze(0)
        Z_rep = Z_rep.expand(
            self.config.uncertainty_n_models,
            bsize,
            self.config.uncertainty_n_pred,
            -1,
        )

        input_images = input_images.unsqueeze(0)
        input_states = input_states.unsqueeze(0)
        actions = actions.unsqueeze(0)
        input_images = input_images.expand(
            self.config.uncertainty_n_models,
            bsize,
            ncond,
            channels,
            height,
            width,
        )
        input_states = input_states.expand(
            self.config.uncertainty_n_models, bsize, ncond, 5
        )
        actions = actions.expand(
            self.config.uncertainty_n_models,
            bsize,
            self.config.uncertainty_n_pred,
            2,
        )
        input_images = input_images.contiguous().view(
            bsize * self.config.uncertainty_n_models,
            ncond,
            channels,
            height,
            width,
        )
        input_states = input_states.contiguous().view(
            bsize * self.config.uncertainty_n_models, ncond, 5
        )
        actions = actions.contiguous().view(
            bsize * self.config.uncertainty_n_models,
            self.config.uncertainty_n_pred,
            2,
        )
        Z_rep = Z_rep.contiguous().view(
            self.config.uncertainty_n_models * bsize,
            self.config.uncertainty_n_pred,
            -1,
        )

        original_value = self.forward_model.training  # to switch back later
        # turn on dropout, for uncertainty estimation
        self.forward_model.train()
        predictions = self.forward_model.unfold(
            actions.clone(),
            dict(
                input_images=input_images,
                input_states=input_states,
            ),
            Z=Z_rep.clone(),
        )
        self.forward_model.train(original_value)

        car_sizes_temp = (
            car_sizes.unsqueeze(0)
            .expand(self.config.uncertainty_n_models, bsize, 2)
            .contiguous()
            .view(self.config.uncertainty_n_models * bsize, 2)
        )
        costs = self.compute_state_costs_for_uncertainty(
            predictions["pred_images"],
            predictions["pred_states"],
            car_sizes_temp,
        )

        pred_costs = (
            self.config.lambda_p * costs["proximity_cost"]
            + self.config.lambda_l * costs["lane_cost"]
            + self.config.lambda_o * costs["offroad_cost"]
        )

        pred_images = predictions["pred_images"].view(
            self.config.uncertainty_n_models,
            bsize,
            self.config.uncertainty_n_pred,
            -1,
        )
        pred_states = predictions["pred_states"].view(
            self.config.uncertainty_n_models,
            bsize,
            self.config.uncertainty_n_pred,
            -1,
        )
        pred_costs = pred_costs.view(
            self.config.uncertainty_n_models,
            bsize,
            self.config.uncertainty_n_pred,
            -1,
        )
        # use variance rather than standard deviation, since it is not
        # differentiable at 0 due to sqrt
        pred_images_var = torch.var(pred_images, 0).mean(2)
        pred_states_var = torch.var(pred_states, 0).mean(2)
        pred_costs_var = torch.var(pred_costs, 0).mean(2)
        pred_costs_mean = torch.mean(pred_costs, 0)
        pred_images = pred_images.view(
            self.config.uncertainty_n_models * bsize,
            self.config.uncertainty_n_pred,
            channels,
            height,
            width,
        )

        pred_states = pred_states.view(
            self.config.uncertainty_n_models * bsize,
            self.config.uncertainty_n_pred,
            5,
        )

        if not estimation:
            # This is the uncertainty loss of different terms together.
            u_loss_costs = torch.relu(
                (pred_costs_var - self.u_costs_mean) / self.u_costs_std
                - self.config.u_hinge
            )
            # u_loss_states = torch.relu(
            #     (pred_states_var - self.u_states_mean) / self.u_states_std
            #     - self.config.u_hinge
            # )
            u_loss_images = torch.relu(
                (pred_images_var - self.u_images_mean) / self.u_images_std
                - self.config.u_hinge
            )
            total_u_loss = (
                u_loss_costs.mean()
                # + u_loss_states.mean()
                + u_loss_images.mean()
            )
        else:
            total_u_loss = None
            # We disabled gradients earlier for the estimation case.
            torch.set_grad_enabled(True)

        return dict(
            pred_images_var=pred_images_var,
            pred_states_var=pred_states_var,
            pred_costs_var=pred_costs_var,
            pred_costs_mean=pred_costs_mean,
            total_u_loss=total_u_loss,
        )

    def calculate_uncertainty_cost(self, inputs, predictions):
        if self.config.u_reg > 0:
            result = self.compute_uncertainty_batch(
                dict(
                    inputs,
                    actions=predictions["pred_actions"],
                    # input_images=inputs['input_images'],
                    # input_states=inputs['input_states'],
                    # car_sizes=inputs['car_sizes'],
                ),
                Z=predictions["Z"],
                estimation=False,
            )["total_u_loss"]
        else:
            result = torch.tensor(0.0)
        return result

    def estimate_uncertainty_stats(self, dataloader):
        """Computes uncertainty estimates for the ground truth actions in the
        training set. This will give us an idea of what normal ranges are using
        actions the forward model was trained on.
        """
        u_images, u_states, u_costs = [], [], []
        data_iter = iter(dataloader)
        for i in range(self.config.uncertainty_n_batches):
            print(
                (
                    f"[estimating normal uncertainty ranges:"
                    f" {i / self.config.uncertainty_n_batches:2.1%}]"
                ),
                end="\r",
            )
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(self.forward_model.device)
            result = self.compute_uncertainty_batch(
                batch=batch,
                estimation=True,
            )
            u_images.append(result["pred_images_var"])
            u_states.append(result["pred_states_var"])
            u_costs.append(result["pred_costs_var"])

        print("[estimating normal uncertainty ranges: 100.0%]")

        u_images = torch.stack(u_images).view(
            -1, self.config.uncertainty_n_pred
        )
        u_states = torch.stack(u_states).view(
            -1, self.config.uncertainty_n_pred
        )
        u_costs = torch.stack(u_costs).view(-1, self.config.uncertainty_n_pred)

        self.u_images_mean = u_images.mean(0)
        self.u_states_mean = u_states.mean(0)
        self.u_costs_mean = u_costs.mean(0)

        self.u_images_std = u_images.std(0)
        self.u_states_std = u_states.std(0)
        self.u_costs_std = u_costs.std(0)

    def compute_state_costs(self, images, states, car_sizes):
        npred = images.size(1)
        gamma_mask = (
            torch.tensor([0.99 ** t for t in range(npred + 1)])
            .cuda()
            .unsqueeze(0)
        )
        artifact_reduced_images = images ** self.config.artifact_power
        proximity_cost = self.compute_proximity_cost(
            artifact_reduced_images,
            states.data,
            car_sizes,
            unnormalize=True,
        )["costs"]
        lane_cost, prox_map_l = self.compute_lane_cost(
            artifact_reduced_images, car_sizes
        )
        offroad_cost = self.compute_offroad_cost(
            artifact_reduced_images, prox_map_l
        )

        lane_loss = torch.mean(lane_cost * gamma_mask[:, :npred])
        offroad_loss = torch.mean(offroad_cost * gamma_mask[:, :npred])
        proximity_loss = torch.mean(proximity_cost * gamma_mask[:, :npred])
        return dict(
            proximity_cost=proximity_cost,
            lane_cost=lane_cost,
            offroad_cost=offroad_cost,
            lane_loss=lane_loss,
            offroad_loss=offroad_loss,
            proximity_loss=proximity_loss,
        )

    def compute_state_costs_for_uncertainty(self, images, states, car_sizes):
        return self.compute_state_costs(images, states, car_sizes)

    def compute_state_costs_for_training(
        self, _inputs, images, states, actions, car_sizes
    ):
        return self.compute_state_costs(images, states, car_sizes)

    def compute_state_costs_for_z(self, images, states, car_sizes):
        return self.compute_state_costs(images, states, car_sizes)

    def compute_combined_loss(
        self,
        proximity_loss,
        uncertainty_loss,
        lane_loss,
        action_loss,
        jerk_loss,
        offroad_loss,
        **_kwargs,
    ):
        return (
            self.config.lambda_p * proximity_loss
            + self.config.u_reg * uncertainty_loss
            + self.config.lambda_l * lane_loss
            + self.config.lambda_a * action_loss
            + self.config.lambda_o * offroad_loss
            + self.config.lambda_j * jerk_loss
        )

    def calculate_cost(self, inputs, predictions):
        u_loss = self.calculate_uncertainty_cost(inputs, predictions)
        if predictions["pred_actions"].shape[1] > 1:
            loss_j = (
                (
                    predictions["pred_actions"][:, 1:]
                    - predictions["pred_actions"][:, :-1]
                )
                .norm(2, 2)
                .pow(2)
                .mean()
            )
        else:
            loss_j = 0.0
        loss_a = (predictions["pred_actions"]).norm(2, 2).pow(2).mean()
        state_losses = self.compute_state_costs_for_training(
            inputs,
            predictions["pred_images"],
            predictions["pred_states"],
            predictions["pred_actions"],
            inputs["car_sizes"],
        )
        result = dict(
            **state_losses,
            uncertainty_loss=u_loss,
            action_loss=loss_a,
            jerk_loss=loss_j,
        )
        result["policy_loss"] = self.compute_combined_loss(**result)
        result["collisions"] = (
            state_losses["collisions"]
            if "collisions" in state_losses
            else None
        )
        return result

    def calculate_z_cost(self, inputs, predictions, original_z=None):
        u_loss = self.calculate_uncertainty_cost(inputs, predictions)
        proximity_loss = self.compute_state_costs_for_z(
            predictions["pred_images"],
            predictions["pred_states"],
            inputs["car_sizes"],
        )["proximity_loss"]
        result = self.compute_combined_loss(
            proximity_loss=-1 * proximity_loss,
            uncertainty_loss=u_loss,
            lane_loss=0,
            action_loss=0,
            jerk_loss=0,
            offroad_loss=0,
        )
        z_reg = torch.tensor(0)
        if original_z is not None:
            z_reg = self.config.dreaming_z_reg * (
                (predictions["Z"] - original_z).norm(2, -1).mean()
                / predictions["Z"].shape[-1]
            )
            result += z_reg
        components = dict(
            proximity_loss=proximity_loss, u_loss=u_loss, z_reg=z_reg
        )
        return result, components

    def get_grad_vid(self, policy_model, batch, device="cuda"):
        input_images = batch["input_images"].clone()
        input_states = batch["input_states"].clone()
        car_sizes = batch["car_sizes"].clone()

        input_images = input_images.clone().float().div_(255.0)
        input_states = self.normalizer.normalize_states(input_states)
        if input_images.dim() == 4:  # if processing single vehicle
            input_images = input_images.to(device).unsqueeze(0)
            input_states = input_states.to(device).unsqueeze(0)
            car_sizes = car_sizes.to(device).unsqueeze(0)

        input_images.requires_grad = True
        input_states.requires_grad = True
        input_images.retain_grad()
        input_states.retain_grad()

        costs = self.compute_state_costs(input_images, input_states, car_sizes)
        combined_loss = self.compute_combined_loss(
            proximity_loss=costs["proximity_loss"],
            uncertainty_loss=0,
            lane_loss=costs["lane_loss"],
            action_loss=0,
            jerk_loss=0,
            offroad_loss=costs["offroad_loss"],
        )
        combined_loss.backward()
        return input_images.grad[:, :, :3].abs().clamp(max=1.0)

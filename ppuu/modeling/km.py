import torch


def predict_states(states, actions, normalizer, timestep=0.1):
    """
    Args:
        - states : tensor of shape [batch size, 4]
        - actions : tensor of shape [batch size, 2]
        - normalizer: normalizer object ot decode normalized values
        - timestep : the time delta between two consecutive states
    """
    states = states.clone()
    actions = actions.clone()

    actions = normalizer.unnormalize_actions(actions)
    states = normalizer.unnormalize_states(states)

    a = actions[:, 0]
    b = actions[:, 1].unsqueeze(1)

    positions = states[:, :2]

    speeds_norm = states[:, 4].unsqueeze(1)

    directions = states[:, 2:4]

    directions = directions / directions.norm(dim=1).unsqueeze(1)

    new_positions = positions + timestep * directions * speeds_norm

    ortho_directions = torch.stack([directions[:, 1], -directions[:, 0]], axis=1)

    new_directions_unnormed = directions + ortho_directions * b * speeds_norm * timestep
    # + torch.tensor([1e-6, 0]).unsqueeze(0).to(directions.device)

    new_directions = new_directions_unnormed / (
        torch.clamp(new_directions_unnormed.norm(dim=1).view(positions.shape[0], 1), min=1e-8, max=1e6,)
    )

    new_speeds_norm = speeds_norm + a.unsqueeze(1) * timestep

    new_states = torch.cat([new_positions, new_directions, new_speeds_norm], 1)
    new_states = normalizer.normalize_states(new_states)

    return new_states


def predict_states_diff(states, actions, normalizer, timestep=0.1):
    """
    Args:
        - states : tensor of shape [batch size, 5]
        - actions : tensor of shape [batch size, 2]
        - normalizer: normalizer object ot decode normalized values
        - timestep : the time delta between two consecutive states
    """
    states = states.clone()
    actions = actions.clone()

    actions = normalizer.unnormalize_actions(actions)
    states = normalizer.unnormalize_states(states)

    # we don't care about x, y coordinates because we're predicting diff
    states[:, :2] = 0

    a = actions[:, 0]
    b = actions[:, 1].unsqueeze(1)

    positions = states[:, :2]

    speeds_norm = states[:, 4].unsqueeze(1)

    directions = states[:, 2:4]

    directions = directions / directions.norm(dim=1).unsqueeze(1)

    new_positions = positions + timestep * directions * speeds_norm

    ortho_directions = torch.stack([directions[:, 1], -directions[:, 0]], axis=1)

    new_directions_unnormed = directions + ortho_directions * b * speeds_norm * timestep
    # + torch.tensor([1e-6, 0]).unsqueeze(0).to(directions.device)

    new_directions = new_directions_unnormed / (
        torch.clamp(new_directions_unnormed.norm(dim=1).view(positions.shape[0], 1), min=1e-12, max=1e6,)
    )

    new_speeds_norm = speeds_norm + a.unsqueeze(1) * timestep

    new_states = torch.cat([new_positions, new_directions, new_speeds_norm], 1)
    new_states = normalizer.normalize_states(new_states)

    return new_states


class StatePredictor:
    def __init__(self, diff, normalizer):
        self.diff = diff
        self.normalizer = normalizer

    def __call__(self, *args, **kwargs):
        assert self.normalizer is not None, "Normalizer is not set for state predictor"
        if self.diff:
            return predict_states(*args, normalizer=self.normalizer, **kwargs)
        else:
            return predict_states_diff(*args, normalizer=self.normalizer, **kwargs)

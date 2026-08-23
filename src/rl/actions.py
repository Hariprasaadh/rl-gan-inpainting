from typing import Dict, Union
import torch
import numpy as np

# Action Space Constants
ACTION_GLOBAL = 0
ACTION_LOCAL = 1
ACTION_BOUNDARY = 2
ACTION_TEXTURE = 3
ACTION_STOP = 4  # Version 2 only

ACTION_NAMES: Dict[int, str] = {
    ACTION_GLOBAL: "Global Completion",
    ACTION_LOCAL: "Local Refinement",
    ACTION_BOUNDARY: "Boundary Refinement",
    ACTION_TEXTURE: "Texture Refinement",
    ACTION_STOP: "Stop",
}

# Action Costs C(a_t)
ACTION_COSTS: Dict[int, float] = {
    ACTION_GLOBAL: 0.00,    # Baseline / free
    ACTION_LOCAL: 0.10,     # Small local refinement computation
    ACTION_BOUNDARY: 0.15,  # Edge/boundary refinement computation
    ACTION_TEXTURE: 0.20,   # Heavy detail/texture computation
    ACTION_STOP: 0.00,      # Stop is cost-free
}


def get_action_name(action: int) -> str:
    """Return human-readable name of action."""
    return ACTION_NAMES.get(int(action), f"Unknown Action ({action})")


def get_action_cost(action: int, custom_costs: Union[Dict[int, float], None] = None) -> float:
    """Return computation cost penalty C(a_t) for action."""
    costs = custom_costs or ACTION_COSTS
    return costs.get(int(action), 0.0)


def decode_action(action: Union[int, np.integer, torch.Tensor], num_actions: int = 4) -> torch.Tensor:
    """Decode integer action to a one-hot representation tensor."""
    if isinstance(action, (np.integer, int)):
        act_idx = int(action)
    elif isinstance(action, torch.Tensor):
        act_idx = int(action.item())
    else:
        act_idx = int(action)

    one_hot = torch.zeros(num_actions, dtype=torch.float32)
    if 0 <= act_idx < num_actions:
        one_hot[act_idx] = 1.0
    return one_hot

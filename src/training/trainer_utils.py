import os
from typing import Optional, Dict, Any
import yaml
import torch
import torch.nn as nn


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Select compute device (cuda if available, else cpu)."""
    if requested_device and requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested_device)
        return torch.device("cpu")
    elif requested_device and requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    state_dict: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
) -> None:
    """Save model checkpoint safely."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state_dict, filepath)
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), "best_model.pt")
        torch.save(state_dict, best_path)


def load_checkpoint(filepath: str, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
    return torch.load(filepath, map_location=device, weights_only=False)

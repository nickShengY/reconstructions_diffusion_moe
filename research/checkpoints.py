import os
import torch
from typing import Any, Dict


def save_state(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_state(path: str, device: str = "cpu") -> Dict[str, Any]:
    if not path:
        raise ValueError("Checkpoint path cannot be empty")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)

"""Device resolution for training workers (CUDA → MPS → CPU)."""

import torch


def resolve_device(preference: str | None = None) -> torch.device:
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_rng_state(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.get_rng_state_all()
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def set_device_rng_state(device: torch.device, state) -> None:
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)

"""Compute device detection: CUDA → MPS → CPU (DESIGN.md §3)."""

import torch


def device_info() -> dict:
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
    if torch.backends.mps.is_available():
        return {
            "device": "mps",
            "name": "Apple Silicon GPU",
            "torch": torch.__version__,
        }
    return {"device": "cpu", "name": "CPU", "torch": torch.__version__}

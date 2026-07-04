"""Checkpoint save/load with the full contents mandated by DESIGN.md §5:
resuming must reproduce *training behavior*, not merely weights.
"""

import random
from pathlib import Path

import torch

from app.training.device import device_rng_state, set_device_rng_state

MNIST_NORMALIZATION = {"mean": [0.1307], "std": [0.3081]}


def save_checkpoint(path: Path, *, run_id: str, config: dict, model, optimizer,
                    epoch: int, global_step: int, best_metric: float,
                    device: torch.device, scheduler=None,
                    arch: str = "simple_mlp", classes: list | None = None,
                    normalization: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,               # fully completed epochs
            "global_step": global_step,
            "best_metric": best_metric,
            "rng": {
                "python": random.getstate(),
                "torch_cpu": torch.get_rng_state(),
                # Cross-device caveat (DESIGN.md §5): this state only restores
                # on the same backend; CUDA↔MPS resume is valid but not
                # bit-identical.
                "device_type": device.type,
                "device": device_rng_state(device),
            },
            "class_mapping": classes if classes is not None else list(range(10)),
            "normalization": normalization or MNIST_NORMALIZATION,
            "config": config,
            "run_id": run_id,
            "parent_run_id": config.get("parent_run_id"),
            "torch_version": torch.__version__,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng(ckpt: dict, device: torch.device) -> None:
    rng = ckpt.get("rng") or {}
    if rng.get("python"):
        random.setstate(tuple(rng["python"]) if isinstance(rng["python"], list) else rng["python"])
    if rng.get("torch_cpu") is not None:
        torch.set_rng_state(rng["torch_cpu"])
    if rng.get("device_type") == device.type:
        set_device_rng_state(device, rng.get("device"))

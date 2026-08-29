"""Training worker — runs in a subprocess (spawn context).

The worker never touches the database (single-writer discipline lives in the
manager); it communicates exclusively by putting event dicts on a
multiprocessing queue and honoring a stop Event: finish the current batch,
save a checkpoint, emit `run_status stopped`, exit (DESIGN.md §5).
"""

import random
import time
import traceback
import uuid
from pathlib import Path

from app.training.checkpoint import (
    MNIST_NORMALIZATION,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from app.training.device import resolve_device
from app.training.models import SimpleMLP

BATCH_EVENT_EVERY = 20  # batches between batch_metrics events (throttling, §4)


def worker_main(run_id: str, config: dict, run_dir: str, queue, stop_event) -> None:
    def emit(type_, payload):
        queue.put({"type": type_, "payload": payload})

    try:
        track = config.get("track")
        if track == "fake":
            fake_run(emit, stop_event, config)
        elif track == "custom_finetune":
            from app.training.finetune import finetune_run
            finetune_run(run_id, config, Path(run_dir), emit, stop_event)
        else:
            mnist_run(run_id, config, Path(run_dir), emit, stop_event)
    except Exception:
        emit("run_status", {"status": "failed", "error": traceback.format_exc()[-2000:]})


# ── fake worker: proves the event backbone without PyTorch (§9 build order) ──

def fake_run(emit, stop_event, config: dict) -> None:
    epochs = config.get("epochs", 2)
    batches = config.get("batches_per_epoch", 5)
    delay = config.get("delay", 0.05)
    emit("run_status", {"status": "running", "device": "fake"})
    step = 0
    for epoch in range(1, epochs + 1):
        for _ in range(batches):
            step += 1
            time.sleep(delay)
            if step % 2 == 0:
                emit("batch_metrics", {"global_step": step, "epoch": epoch,
                                       "loss": 1.0 / step, "lr": 0.1})
            if stop_event.is_set():
                # Non-empty path so resume validation passes in lifecycle
                # tests; the fake track never loads it.
                emit("checkpoint_saved", {"checkpoint_id": "fake", "epoch": epoch - 1,
                                          "global_step": step, "path": "fake-checkpoint"})
                emit("run_status", {"status": "stopped"})
                return
        emit("epoch_metrics", {"epoch": epoch, "train_loss": 1.0 / epoch,
                               "train_acc": 1 - 0.5 / epoch, "val_loss": 1.1 / epoch,
                               "val_acc": 1 - 0.6 / epoch, "lr": 0.1})
    emit("run_status", {"status": "done"})


# ── real MNIST training ──────────────────────────────────────────────────────

def mnist_run(run_id: str, config: dict, run_dir: Path, emit, stop_event) -> None:
    import torch
    from torch import nn
    from torchvision import datasets, transforms

    from app.config import DATA_DIR

    device = resolve_device(config.get("device"))
    seed = config.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)

    hidden = tuple(config.get("hidden", [128, 64]))
    model = SimpleMLP(hidden=hidden).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.get("lr", 0.1),
                                momentum=config.get("momentum", 0.9))
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, global_step, best_metric = 0, 0, 0.0
    resume_path = config.get("resume_checkpoint")
    if resume_path:
        ckpt = load_checkpoint(resume_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # v1 simplification: a mid-epoch checkpoint re-runs its partial epoch
        # (restoring a DataLoader's position isn't worth the machinery; the
        # cross-device RNG caveat already rules out bit-identical resume).
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_metric = ckpt.get("best_metric", 0.0)
        restore_rng(ckpt, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_NORMALIZATION["mean"], MNIST_NORMALIZATION["std"]),
    ])
    cache = str(DATA_DIR / "cache")
    train_ds = datasets.MNIST(cache, train=True, download=True, transform=transform)
    val_ds = datasets.MNIST(cache, train=False, download=True, transform=transform)
    batch_size = config.get("batch_size", 128)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512)

    emit("run_status", {"status": "running", "device": device.type,
                        "params": sum(p.numel() for p in model.parameters())})

    def checkpoint(epoch_completed: int) -> None:
        ckpt_id = uuid.uuid4().hex[:12]
        path = run_dir / f"ckpt_{ckpt_id}.pt"
        save_checkpoint(path, run_id=run_id, config=config, model=model,
                        optimizer=optimizer, epoch=epoch_completed,
                        global_step=global_step, best_metric=best_metric, device=device)
        emit("checkpoint_saved", {"checkpoint_id": ckpt_id, "epoch": epoch_completed,
                                  "global_step": global_step, "path": str(path),
                                  "best_metric": best_metric})

    epochs = config.get("epochs", 3)
    lr = optimizer.param_groups[0]["lr"]
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        seen, correct, loss_sum = 0, 0, 0.0
        grad_norms: dict[str, list[float]] = {}
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            batch_n = y.size(0)
            seen += batch_n
            loss_sum += loss.item() * batch_n
            correct += (logits.argmax(dim=1) == y).sum().item()

            if global_step % BATCH_EVENT_EVERY == 0:
                emit("batch_metrics", {"global_step": global_step, "epoch": epoch,
                                       "loss": round(loss.item(), 5), "lr": lr})
                for name, layer in model.linear_layers():
                    if layer.weight.grad is not None:
                        grad_norms.setdefault(name, []).append(
                            layer.weight.grad.norm().item())

            if stop_event.is_set():
                checkpoint(epoch_completed=epoch - 1)
                emit("run_status", {"status": "stopped"})
                return

        val_loss, val_acc, dead_pct = _validate(model, val_loader, loss_fn, device)
        best_metric = max(best_metric, val_acc)
        emit("epoch_metrics", {"epoch": epoch,
                               "train_loss": round(loss_sum / seen, 5),
                               "train_acc": round(correct / seen, 5),
                               "val_loss": round(val_loss, 5),
                               "val_acc": round(val_acc, 5), "lr": lr})
        emit("layer_stats", {"epoch": epoch, "layers": [
            {"name": name,
             "shape": list(layer.weight.shape),
             "grad_norm": round(sum(g) / len(g), 5) if (g := grad_norms.get(name)) else None,
             "weight_norm": round(layer.weight.norm().item(), 5),
             "dead_relu_pct": dead_pct.get(name)}
            for name, layer in model.linear_layers()
        ]})
        checkpoint(epoch_completed=epoch)

    emit("run_status", {"status": "done", "best_metric": round(best_metric, 5)})


def _validate(model, val_loader, loss_fn, device):
    """Validation pass; also measures dead-ReLU % per hidden layer (units that
    never activate across the whole validation set)."""
    import torch

    model.eval()
    ever_active: dict[str, torch.Tensor] = {}
    hooks = []
    for name, relu in model.relu_layers():
        def hook(_m, _in, out, name=name):
            active = (out > 0).any(dim=0)
            prev = ever_active.get(name)
            ever_active[name] = active if prev is None else (prev | active)
        hooks.append(relu.register_forward_hook(hook))

    seen, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.size(0)
    for h in hooks:
        h.remove()

    # A ReLU at net.{i} follows the Linear at net.{i-1}; report deadness
    # against that Linear layer's name so the UI can join the two.
    dead_pct = {}
    for name, ever in ever_active.items():
        idx = int(name.split(".")[1])
        linear_name = f"net.{idx - 1}"
        dead_pct[linear_name] = round(100.0 * (~ever).sum().item() / ever.numel(), 2)
    return loss_sum / seen, correct / seen, dead_pct

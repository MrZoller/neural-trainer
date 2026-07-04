"""Track 2 worker: fine-tune a pretrained MobileNetV3-Small head on a frozen
dataset manifest (DESIGN.md §4).

The backbone stays frozen (requires_grad=False): those weights already know
edges, textures, and shapes from ImageNet's 1.2M images, which is exactly why
~30-100 photos per class is enough here when training from scratch would need
thousands. Only the classifier head learns your classes.
"""

import json
import os
import random
import uuid
from pathlib import Path

from app.config import DATA_DIR

IMAGENET_NORM = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# Keep pretrained-weight downloads inside data/ so the app stays portable.
os.environ.setdefault("TORCH_HOME", str(DATA_DIR / "cache" / "torch"))


def build_model(num_classes: int):
    from torchvision import models

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False
    # Replace the final Linear with one sized for the user's classes; the
    # whole (small) classifier head trains.
    import torch.nn as nn
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


class ManifestDataset:
    def __init__(self, entries: list[dict], class_to_idx: dict[str, int], transform):
        self.entries = entries
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from PIL import Image

        e = self.entries[idx]
        img = Image.open(e["path"]).convert("RGB")
        return self.transform(img), self.class_to_idx[e["label"]]


def finetune_run(run_id: str, config: dict, run_dir: Path, emit, stop_event) -> None:
    import torch
    from torch import nn
    from torchvision import transforms

    from app.training.checkpoint import load_checkpoint, restore_rng, save_checkpoint
    from app.training.device import resolve_device

    device = resolve_device(config.get("device"))
    seed = config.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)

    manifest_doc = json.loads(Path(config["manifest_path"]).read_text())
    classes = manifest_doc["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    entries = manifest_doc["manifest"]
    train_entries = [e for e in entries if e["split"] == "train"]
    val_entries = [e for e in entries if e["split"] == "val"]

    model = build_model(len(classes)).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.get("lr", 1e-3))
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, global_step, best_metric = 0, 0, 0.0
    if config.get("resume_checkpoint"):
        ckpt = load_checkpoint(config["resume_checkpoint"])
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch, global_step = ckpt["epoch"], ckpt["global_step"]
        best_metric = ckpt.get("best_metric", 0.0)
        restore_rng(ckpt, device)

    # Augmentation gives each photo many "free" variants; val sees the clean
    # deterministic pipeline only.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_NORM["mean"], IMAGENET_NORM["std"]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_NORM["mean"], IMAGENET_NORM["std"]),
    ])
    batch_size = config.get("batch_size", 32)
    train_loader = torch.utils.data.DataLoader(
        ManifestDataset(train_entries, class_to_idx, train_tf),
        batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        ManifestDataset(val_entries, class_to_idx, val_tf), batch_size=64)

    frozen_params = sum(p.numel() for p in model.features.parameters())
    head_params = sum(p.numel() for p in trainable)
    emit("run_status", {"status": "running", "device": device.type,
                        "params": head_params, "frozen_params": frozen_params,
                        "classes": classes,
                        "n_train": len(train_entries), "n_val": len(val_entries)})

    # Tiny datasets have few batches per epoch — adapt event cadence so the
    # batch-loss chart still gets points.
    emit_every = max(1, min(20, len(train_loader) // 5 or 1))
    lr = optimizer.param_groups[0]["lr"]

    def checkpoint(epoch_completed: int) -> None:
        ckpt_id = uuid.uuid4().hex[:12]
        path = run_dir / f"ckpt_{ckpt_id}.pt"
        save_checkpoint(path, run_id=run_id, config=config, model=model,
                        optimizer=optimizer, epoch=epoch_completed,
                        global_step=global_step, best_metric=best_metric,
                        device=device, arch="mobilenet_v3_small",
                        classes=classes, normalization=IMAGENET_NORM)
        emit("checkpoint_saved", {"checkpoint_id": ckpt_id, "epoch": epoch_completed,
                                  "global_step": global_step, "path": str(path),
                                  "best_metric": best_metric})

    epochs = config.get("epochs", 10)
    head_linears = [(f"classifier.{i}", m) for i, m in enumerate(model.classifier)
                    if isinstance(m, nn.Linear)]
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
            seen += y.size(0)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            if global_step % emit_every == 0:
                emit("batch_metrics", {"global_step": global_step, "epoch": epoch,
                                       "loss": round(loss.item(), 5), "lr": lr})
                for name, layer in head_linears:
                    if layer.weight.grad is not None:
                        grad_norms.setdefault(name, []).append(
                            layer.weight.grad.norm().item())
            if stop_event.is_set():
                checkpoint(epoch_completed=epoch - 1)
                emit("run_status", {"status": "stopped"})
                return

        val_loss, val_acc = _validate(model, val_loader, loss_fn, device)
        best_metric = max(best_metric, val_acc)
        emit("epoch_metrics", {"epoch": epoch,
                               "train_loss": round(loss_sum / seen, 5),
                               "train_acc": round(correct / seen, 5),
                               "val_loss": round(val_loss, 5),
                               "val_acc": round(val_acc, 5), "lr": lr})
        emit("layer_stats", {"epoch": epoch, "layers": [
            {"name": f"features [frozen — {frozen_params:,} ImageNet params]",
             "shape": None, "grad_norm": None, "weight_norm": None,
             "dead_relu_pct": None},
            *({"name": name, "shape": list(layer.weight.shape),
               "grad_norm": round(sum(g) / len(g), 5) if (g := grad_norms.get(name)) else None,
               "weight_norm": round(layer.weight.norm().item(), 5),
               "dead_relu_pct": None}
              for name, layer in head_linears),
        ]})
        checkpoint(epoch_completed=epoch)

    emit("run_status", {"status": "done", "best_metric": round(best_metric, 5)})


def _validate(model, val_loader, loss_fn, device):
    import torch

    model.eval()
    seen, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.size(0)
    if seen == 0:
        return 0.0, 0.0
    return loss_sum / seen, correct / seen

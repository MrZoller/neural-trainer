"""Dataset versioning (DESIGN.md §3): a run trains on a frozen manifest, so
run history stays meaningful when the living dataset changes.
"""

import hashlib
import json
import uuid
from pathlib import Path

from app.datasets.split import assign_split
from app.store.db import Database


class FreezeError(ValueError):
    pass


def _eligible(db: Database, dataset_id: str) -> list[dict]:
    return [i for i in db.list_images(dataset_id)
            if i["label"] and not i["excluded"]]


def build_report(db: Database, dataset_id: str) -> dict:
    """Quality report (§8): component-level, no composite score. Includes a
    dry-run of the split so warnings match what freezing would produce."""
    all_images = db.list_images(dataset_id)
    images = [i for i in all_images if i["label"] and not i["excluded"]]
    split, warnings = assign_split(images)

    classes: dict[str, dict] = {}
    for img in images:
        c = classes.setdefault(img["label"], {"images": 0, "groups": set(),
                                              "train": 0, "val": 0})
        c["images"] += 1
        c["groups"].add(img["capture_group"])
        c[split[img["id"]]] += 1

    return {
        "n_images": len(all_images),
        "n_labeled": len(images),
        "n_unlabeled": len(all_images) - len(images),
        "classes": {label: {**c, "groups": len(c["groups"])}
                    for label, c in sorted(classes.items())},
        "warnings": warnings,
    }


def freeze_version(db: Database, dataset_id: str) -> dict:
    """Snapshot the current labeled images into an immutable manifest with a
    group-aware split. Returns the stored version row."""
    images = _eligible(db, dataset_id)
    classes = sorted({i["label"] for i in images})
    if len(classes) < 2:
        raise FreezeError("need at least 2 labeled classes to train a classifier")
    counts = {c: sum(1 for i in images if i["label"] == c) for c in classes}
    thin = [c for c, n in counts.items() if n < 4]
    if thin:
        raise FreezeError(f"classes {thin} have fewer than 4 images — add more first")

    split, warnings = assign_split(images)
    manifest = sorted(
        ({"image_id": i["id"], "content_hash": i["content_hash"], "path": i["path"],
          "label": i["label"], "capture_group": i["capture_group"],
          "split": split[i["id"]]} for i in images),
        key=lambda e: e["content_hash"])
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]

    stats = {
        "counts": counts,
        "train": sum(1 for e in manifest if e["split"] == "train"),
        "val": sum(1 for e in manifest if e["split"] == "val"),
        "warnings": warnings,
    }
    version_id = uuid.uuid4().hex[:12]
    return db.create_dataset_version(version_id, dataset_id, manifest,
                                     manifest_hash, classes, stats)


def write_manifest_file(version: dict, run_dir: Path) -> Path:
    """Copy of the frozen manifest alongside the run's checkpoints
    (DESIGN.md §3: runs/<id>/manifest.json)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(
        {"dataset_version_id": version["id"], "manifest_hash": version["manifest_hash"],
         "classes": version["classes"], "stats": version["stats"],
         "manifest": version["manifest"]}, indent=1))
    return path

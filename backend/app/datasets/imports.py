"""Folder import: content-addressed storage + capture-group inference.

Capture groups exist so near-duplicate photos never straddle the train/val
split (DESIGN.md §8). For imports the groups are inferred from EXIF
timestamps: photos in the same folder taken within BURST_WINDOW seconds of
each other form one group; images without EXIF fall back to per-file groups.
Grouping is recomputed dataset-wide after every import batch (so bursts that
arrive across multiple upload requests still merge), but never touches
groups a user set by hand (group_source = 'manual').
"""

import hashlib
import io
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from app.store.db import Database

BURST_WINDOW = timedelta(seconds=10)
ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024

EXIF_DATETIME_ORIGINAL = 36867
EXIF_IFD = 0x8769
EXIF_DATETIME = 306


class ImportError_(ValueError):
    pass


def exif_taken_at(img: Image.Image) -> str | None:
    try:
        exif = img.getexif()
        raw = exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
        if not raw:
            return None
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").isoformat()
    except Exception:
        return None


def import_file(db: Database, dataset_id: str, files_root: Path, *,
                rel_name: str, content_type: str, data: bytes,
                default_label: str | None = None) -> str:
    """Store one uploaded file. Returns 'added' | 'duplicate'. Raises
    ImportError_ for files that fail the §3 upload limits."""
    if content_type not in ALLOWED_TYPES:
        raise ImportError_(f"{rel_name}: type {content_type} not allowed (jpeg/png/webp)")
    if len(data) > MAX_FILE_BYTES:
        raise ImportError_(f"{rel_name}: exceeds {MAX_FILE_BYTES // (1024*1024)}MB limit")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))  # verify() invalidates the object
    except Exception:
        raise ImportError_(f"{rel_name}: not a decodable image")

    rel = Path(rel_name.replace("\\", "/"))
    folder = rel.parent.as_posix() if rel.parent.as_posix() != "." else ""
    # Folder import auto-labels from the containing folder name (cat/IMG.jpg
    # → "cat"); the labeling grid can always override.
    label = default_label or (rel.parent.name or None)

    content_hash = hashlib.sha256(data).hexdigest()[:16]
    path = files_root / dataset_id / f"{content_hash}{ALLOWED_TYPES[content_type]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)

    image_id = uuid.uuid4().hex[:12]
    added = db.add_image({
        "id": image_id,
        "dataset_id": dataset_id,
        "content_hash": content_hash,
        "path": str(path),
        "filename": rel.name,
        "folder": folder,
        "label": label,
        "capture_group": image_id,  # provisional; recompute_groups follows
        "group_source": "file",
        "taken_at": exif_taken_at(img),
    })
    return "added" if added else "duplicate"


def recompute_groups(db: Database, dataset_id: str) -> None:
    """Chain images in the same folder whose EXIF timestamps are within
    BURST_WINDOW of the previous image into one capture group."""
    images = [i for i in db.list_images(dataset_id) if i["group_source"] != "manual"]
    assignments: dict[str, str] = {}

    by_folder: dict[str, list[dict]] = {}
    for img in images:
        if img["taken_at"]:
            by_folder.setdefault(img["folder"], []).append(img)
        else:
            assignments[img["id"]] = img["id"]  # per-file fallback

    for folder_images in by_folder.values():
        folder_images.sort(key=lambda i: (i["taken_at"], i["id"]))
        group_id, prev_t = None, None
        for img in folder_images:
            t = datetime.fromisoformat(img["taken_at"])
            if group_id is None or prev_t is None or t - prev_t > BURST_WINDOW:
                group_id = f"burst-{img['id']}"
            assignments[img["id"]] = group_id
            prev_t = t

    db.set_capture_groups({
        image_id: (group, "exif" if group.startswith("burst-") else "file")
        for image_id, group in assignments.items()
    })

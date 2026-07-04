"""Capture grouping, group-aware split, versioning, dataset API (DESIGN.md §8)."""

import io

import pytest
from PIL import Image

from app.datasets.split import assign_split
from app.store.db import Database


def img_row(i, label, group, taken_at=None, folder=""):
    return {"id": f"i{i}", "dataset_id": "d", "content_hash": f"h{i}",
            "path": f"/x/{i}.jpg", "filename": f"{i}.jpg", "folder": folder,
            "label": label, "capture_group": group, "group_source": "file",
            "taken_at": taken_at, "excluded": 0}


# ── split ─────────────────────────────────────────────────────────────────────

def test_groups_never_straddle_split():
    images = [img_row(i, "cat", f"g{i // 5}") for i in range(20)]  # 4 groups of 5
    images += [img_row(100 + i, "dog", f"h{i // 5}") for i in range(20)]
    split, _ = assign_split(images)
    for img in images:
        group_splits = {split[x["id"]] for x in images
                        if x["capture_group"] == img["capture_group"]}
        assert len(group_splits) == 1  # whole group on one side
    for label in ("cat", "dog"):
        sides = {split[i["id"]] for i in images if i["label"] == label}
        assert sides == {"train", "val"}  # both sides populated per class


def test_single_group_class_goes_to_train_with_warning():
    images = [img_row(i, "cat", "only-group") for i in range(10)]
    images += [img_row(100 + i, "dog", f"g{i}") for i in range(10)]
    split, warnings = assign_split(images)
    assert all(split[f"i{i}"] == "train" for i in range(10))
    assert any("one capture session" in w for w in warnings)


def test_largest_group_stays_in_train():
    images = [img_row(i, "cat", "big") for i in range(50)]
    images += [img_row(100, "cat", "small-a"), img_row(101, "cat", "small-b")]
    split, _ = assign_split(images)
    assert all(split[f"i{i}"] == "train" for i in range(50))


# ── EXIF burst grouping ───────────────────────────────────────────────────────

def test_burst_grouping_chains_within_window(tmp_path):
    from app.datasets.imports import recompute_groups

    db = Database(tmp_path / "t.db")
    db.create_dataset("d", "test")
    times = ["2026-07-01T10:00:00", "2026-07-01T10:00:04", "2026-07-01T10:00:09",
             "2026-07-01T10:05:00",  # > 10s gap → new group
             None]                   # no EXIF → per-file group
    for i, t in enumerate(times):
        db.add_image(img_row(i, "cat", f"i{i}", taken_at=t, folder="cat"))
    recompute_groups(db, "d")

    groups = {i["id"]: i["capture_group"] for i in db.list_images("d")}
    assert groups["i0"] == groups["i1"] == groups["i2"]  # chained burst
    assert groups["i3"] != groups["i0"]
    assert groups["i4"] == "i4"  # per-file fallback

    sources = {i["id"]: i["group_source"] for i in db.list_images("d")}
    assert sources["i0"] == "exif" and sources["i4"] == "file"


def test_manual_groups_survive_recompute(tmp_path):
    from app.datasets.imports import recompute_groups

    db = Database(tmp_path / "t.db")
    db.create_dataset("d", "test")
    db.add_image(img_row(0, "cat", "x", taken_at="2026-07-01T10:00:00", folder="cat"))
    db.update_image("i0", {"capture_group": "my-session", "group_source": "manual"})
    recompute_groups(db, "d")
    assert db.get_image("i0")["capture_group"] == "my-session"


# ── dataset API ───────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NT_DATA_DIR", str(tmp_path))
    import importlib

    from app import config
    importlib.reload(config)
    from app import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


def png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_labels_from_folder_and_dedupes(client):
    ds = client.post("/api/datasets", json={"name": "pets"}).json()
    files = [
        ("files", ("cat/a.png", png_bytes((255, 0, 0)), "image/png")),
        ("files", ("cat/b.png", png_bytes((250, 0, 0)), "image/png")),
        ("files", ("dog/c.png", png_bytes((0, 0, 255)), "image/png")),
        ("files", ("dog/dup.png", png_bytes((0, 0, 255)), "image/png")),  # same bytes as c
        ("files", ("dog/nope.txt", b"not an image", "text/plain")),
    ]
    r = client.post(f"/api/datasets/{ds['id']}/images", files=files).json()
    assert (r["added"], r["duplicates"], len(r["rejected"])) == (3, 1, 1)

    images = client.get(f"/api/datasets/{ds['id']}/images").json()
    assert {i["label"] for i in images} == {"cat", "dog"}
    assert all(i["capture_group"] == i["id"] for i in images)  # no EXIF → per-file

    # thumbnail serving
    file_r = client.get(f"/api/images/{images[0]['id']}/file")
    assert file_r.status_code == 200

    # relabel + exclude via PATCH
    patched = client.patch(f"/api/images/{images[0]['id']}",
                           json={"label": "hamster", "excluded": True}).json()
    assert patched["label"] == "hamster" and patched["excluded"] == 1


def test_report_and_freeze_guards(client):
    ds = client.post("/api/datasets", json={"name": "thin"}).json()
    files = [("files", (f"cat/{i}.png", png_bytes((i * 10, 0, 0)), "image/png"))
             for i in range(5)]
    client.post(f"/api/datasets/{ds['id']}/images", files=files)

    report = client.get(f"/api/datasets/{ds['id']}/report").json()
    assert report["classes"]["cat"]["images"] == 5
    assert any("only 5 images" in w for w in report["warnings"])

    # one class → freeze must refuse
    r = client.post("/api/runs", json={"track": "custom_finetune",
                                       "dataset_id": ds["id"]})
    assert r.status_code == 409
    assert "2 labeled classes" in r.json()["detail"]

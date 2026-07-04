"""Run lifecycle through the full API with the fake worker (DESIGN.md §5, §9:
prove the event backbone before PyTorch enters the picture)."""

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NT_DATA_DIR", str(tmp_path))
    # Re-import config-dependent modules fresh so DATA_DIR override applies.
    import importlib

    from app import config
    importlib.reload(config)
    from app import main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def wait_for_status(client, run_id, statuses, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{run_id}").json()["status"]
        if status in statuses:
            return status
        time.sleep(0.1)
    raise AssertionError(f"run never reached {statuses}, last status: {status}")


def test_fake_run_completes_with_full_event_log(client):
    run = client.post("/api/runs", json={"track": "fake", "epochs": 2}).json()
    assert wait_for_status(client, run["id"], {"done"}) == "done"

    events = client.get(f"/api/runs/{run['id']}/events").json()
    types = [e["type"] for e in events]
    statuses = [e["payload"]["status"] for e in events if e["type"] == "run_status"]
    assert statuses == ["queued", "running", "done"]
    assert "batch_metrics" in types and "epoch_metrics" in types
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


def test_replay_after_seq_over_rest(client):
    run = client.post("/api/runs", json={"track": "fake", "epochs": 1}).json()
    wait_for_status(client, run["id"], {"done"})
    all_events = client.get(f"/api/runs/{run['id']}/events").json()
    tail = client.get(f"/api/runs/{run['id']}/events?after_seq=2").json()
    assert tail == all_events[2:]


def test_stop_is_graceful(client):
    run = client.post("/api/runs", json={
        "track": "fake", "epochs": 50, "batches_per_epoch": 20, "delay": 0.05,
    }).json()
    wait_for_status(client, run["id"], {"running"})
    r = client.post(f"/api/runs/{run['id']}/actions", json={"action": "stop"})
    assert r.json()["status"] == "stopping"
    assert wait_for_status(client, run["id"], {"stopped"}) == "stopped"
    statuses = [e["payload"]["status"]
                for e in client.get(f"/api/runs/{run['id']}/events").json()
                if e["type"] == "run_status"]
    assert statuses == ["queued", "running", "stopping", "stopped"]


def test_kill_queued_run_is_cancelled(client):
    blocker = client.post("/api/runs", json={
        "track": "fake", "epochs": 50, "batches_per_epoch": 20, "delay": 0.05,
    }).json()
    wait_for_status(client, blocker["id"], {"running"})
    queued = client.post("/api/runs", json={"track": "fake"}).json()
    assert client.get(f"/api/runs/{queued['id']}").json()["status"] == "queued"

    r = client.post(f"/api/runs/{queued['id']}/actions", json={"action": "kill"})
    assert r.json()["status"] == "cancelled"
    client.post(f"/api/runs/{blocker['id']}/actions", json={"action": "kill"})
    wait_for_status(client, blocker["id"], {"killed"})


def test_resume_preserves_parent_config(client):
    """Regression: RunCreate defaults must not clobber parent config on resume
    (epochs=3 default once overwrote a parent's epochs=8)."""
    run = client.post("/api/runs", json={
        "track": "fake", "epochs": 50, "batches_per_epoch": 20, "delay": 0.05,
    }).json()
    wait_for_status(client, run["id"], {"running"})
    client.post(f"/api/runs/{run['id']}/actions", json={"action": "stop"})
    wait_for_status(client, run["id"], {"stopped"})

    child = client.post("/api/runs", json={"parent_run_id": run["id"]}).json()
    assert child["config"]["epochs"] == 50  # parent's value, not the default 3
    assert child["config"]["resume_checkpoint"] == "fake-checkpoint"
    assert child["parent_run_id"] == run["id"]

    explicit = client.post("/api/runs", json={
        "parent_run_id": run["id"], "epochs": 60,
    }).json()
    assert explicit["config"]["epochs"] == 60  # explicit override still works

    for r in (child, explicit):
        client.post(f"/api/runs/{r['id']}/actions", json={"action": "kill"})


def test_action_on_terminal_run_is_rejected(client):
    run = client.post("/api/runs", json={"track": "fake", "epochs": 1}).json()
    wait_for_status(client, run["id"], {"done"})
    r = client.post(f"/api/runs/{run['id']}/actions", json={"action": "stop"})
    assert r.status_code == 409

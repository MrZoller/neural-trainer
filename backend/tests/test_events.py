"""Event store: append/replay semantics (DESIGN.md §4)."""

import sqlite3

import pytest

from app.store.db import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


def test_seq_is_monotonic_per_run(db):
    db.create_run("r1", {}, 1)
    db.create_run("r2", {}, 1)
    e1 = db.append_event("r1", "run_status", {"status": "queued"})
    e2 = db.append_event("r1", "batch_metrics", {"loss": 1.0})
    other = db.append_event("r2", "run_status", {"status": "queued"})
    assert (e1["seq"], e2["seq"]) == (1, 2)
    assert other["seq"] == 1  # per-run, not global


def test_replay_after_seq(db):
    db.create_run("r1", {}, 1)
    for i in range(5):
        db.append_event("r1", "batch_metrics", {"step": i})
    replay = db.events_after("r1", after_seq=3)
    assert [e["seq"] for e in replay] == [4, 5]
    assert [e["payload"]["step"] for e in replay] == [3, 4]


def test_duplicate_seq_rejected(db):
    db.create_run("r1", {}, 1)
    db.append_event("r1", "run_status", {"status": "queued"})
    with pytest.raises(sqlite3.IntegrityError):
        db._conn.execute(
            "INSERT INTO events (run_id, seq, ts, type, payload_json)"
            " VALUES ('r1', 1, '', 'x', '{}')")

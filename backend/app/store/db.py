"""SQLite store: runs, events, checkpoints.

Single-writer discipline (DESIGN.md §4): every write goes through this class,
serialized by a lock. Events are the source of truth; runs.status is a
materialized view of the latest run_status event.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL,
  config_json TEXT NOT NULL,
  parent_run_id TEXT,
  stream_schema_version INTEGER NOT NULL,
  error TEXT
);

CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  ts TEXT NOT NULL,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  UNIQUE(run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);

CREATE TABLE IF NOT EXISTS checkpoints (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  epoch INTEGER NOT NULL,
  global_step INTEGER NOT NULL,
  path TEXT NOT NULL,
  created_at TEXT NOT NULL,
  metrics_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
"""

TERMINAL_STATUSES = {"cancelled", "stopped", "killed", "done", "failed", "interrupted"}
RESUMABLE_STATUSES = {"stopped", "killed", "failed", "interrupted"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    # ── runs ──────────────────────────────────────────────────────────────

    def create_run(self, run_id: str, config: dict, stream_schema_version: int,
                   parent_run_id: str | None = None) -> dict:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs (id, created_at, status, config_json, parent_run_id,"
                " stream_schema_version) VALUES (?, ?, 'queued', ?, ?, ?)",
                (run_id, _now(), json.dumps(config), parent_run_id, stream_schema_version),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._run_dict(row) if row else None

    def list_runs(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [self._run_dict(r) for r in rows]

    def set_run_status(self, run_id: str, status: str, error: str | None = None):
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET status = ?, error = COALESCE(?, error) WHERE id = ?",
                (status, error, run_id),
            )

    def runs_with_status(self, statuses: list[str]) -> list[dict]:
        q = ",".join("?" * len(statuses))
        rows = self._conn.execute(
            f"SELECT * FROM runs WHERE status IN ({q}) ORDER BY created_at", statuses
        ).fetchall()
        return [self._run_dict(r) for r in rows]

    @staticmethod
    def _run_dict(row) -> dict:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json"))
        return d

    # ── events ────────────────────────────────────────────────────────────

    def append_event(self, run_id: str, type_: str, payload: dict) -> dict:
        with self._lock, self._conn:
            (max_seq,) = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id = ?", (run_id,)
            ).fetchone()
            seq = max_seq + 1
            ts = _now()
            self._conn.execute(
                "INSERT INTO events (run_id, seq, ts, type, payload_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, seq, ts, type_, json.dumps(payload)),
            )
        return {"run_id": run_id, "seq": seq, "ts": ts, "type": type_, "payload": payload}

    def events_after(self, run_id: str, after_seq: int = 0, limit: int | None = None) -> list[dict]:
        sql = ("SELECT run_id, seq, ts, type, payload_json FROM events"
               " WHERE run_id = ? AND seq > ? ORDER BY seq")
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql, (run_id, after_seq)).fetchall()
        return [
            {"run_id": r["run_id"], "seq": r["seq"], "ts": r["ts"], "type": r["type"],
             "payload": json.loads(r["payload_json"])}
            for r in rows
        ]

    # ── checkpoints ───────────────────────────────────────────────────────

    def add_checkpoint(self, ckpt_id: str, run_id: str, epoch: int, global_step: int,
                       path: str, metrics: dict | None = None):
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO checkpoints (id, run_id, epoch, global_step,"
                " path, created_at, metrics_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ckpt_id, run_id, epoch, global_step, path, _now(),
                 json.dumps(metrics or {})),
            )

    def latest_checkpoint(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE run_id = ?"
            " ORDER BY global_step DESC, created_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
        return d

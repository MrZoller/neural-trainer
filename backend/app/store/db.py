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

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  path TEXT NOT NULL,
  filename TEXT NOT NULL,
  folder TEXT NOT NULL DEFAULT '',
  label TEXT,
  capture_group TEXT NOT NULL,
  group_source TEXT NOT NULL DEFAULT 'file',  -- exif | file | manual
  excluded INTEGER NOT NULL DEFAULT 0,
  taken_at TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(dataset_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset_id);

CREATE TABLE IF NOT EXISTS dataset_versions (
  id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  manifest_hash TEXT NOT NULL,
  classes_json TEXT NOT NULL,
  stats_json TEXT NOT NULL
);
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
            # Migration for pre-2A databases.
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(runs)")}
            if "dataset_version_id" not in cols:
                self._conn.execute("ALTER TABLE runs ADD COLUMN dataset_version_id TEXT")

    # ── runs ──────────────────────────────────────────────────────────────

    def create_run(self, run_id: str, config: dict, stream_schema_version: int,
                   parent_run_id: str | None = None,
                   dataset_version_id: str | None = None) -> dict:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs (id, created_at, status, config_json, parent_run_id,"
                " stream_schema_version, dataset_version_id) VALUES (?, ?, 'queued', ?, ?, ?, ?)",
                (run_id, _now(), json.dumps(config), parent_run_id,
                 stream_schema_version, dataset_version_id),
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

    # ── datasets ──────────────────────────────────────────────────────────

    def create_dataset(self, dataset_id: str, name: str) -> dict:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO datasets (id, name, created_at) VALUES (?, ?, ?)",
                (dataset_id, name, _now()))
        return self.get_dataset(dataset_id)

    def get_dataset(self, dataset_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None

    def list_datasets(self) -> list[dict]:
        rows = self._conn.execute("""
            SELECT d.*,
                   COUNT(i.id) AS n_images,
                   SUM(CASE WHEN i.label IS NOT NULL AND i.excluded = 0 THEN 1 ELSE 0 END) AS n_labeled
            FROM datasets d LEFT JOIN images i ON i.dataset_id = d.id
            GROUP BY d.id ORDER BY d.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def add_image(self, image: dict) -> bool:
        """Insert an image row; returns False when the content hash already
        exists in this dataset (exact-duplicate upload)."""
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO images (id, dataset_id, content_hash, path, filename,"
                    " folder, label, capture_group, group_source, taken_at, created_at)"
                    " VALUES (:id, :dataset_id, :content_hash, :path, :filename,"
                    " :folder, :label, :capture_group, :group_source, :taken_at, :created_at)",
                    {**image, "created_at": _now()})
            return True
        except sqlite3.IntegrityError:
            return False

    def list_images(self, dataset_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM images WHERE dataset_id = ? ORDER BY folder, taken_at, filename",
            (dataset_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_image(self, image_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return dict(row) if row else None

    def update_image(self, image_id: str, fields: dict) -> None:
        allowed = {"label", "excluded", "capture_group", "group_source"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        assign = ", ".join(f"{k} = ?" for k in sets)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE images SET {assign} WHERE id = ?", (*sets.values(), image_id))

    def set_capture_groups(self, assignments: dict[str, tuple[str, str]]) -> None:
        """Bulk (group, source) reassignment for non-manual images
        (import-time inference)."""
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE images SET capture_group = ?, group_source = ?"
                " WHERE id = ? AND group_source != 'manual'",
                [(g, s, i) for i, (g, s) in assignments.items()])

    # ── dataset versions ──────────────────────────────────────────────────

    def create_dataset_version(self, version_id: str, dataset_id: str, manifest: list,
                               manifest_hash: str, classes: list, stats: dict) -> dict:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO dataset_versions (id, dataset_id, created_at, manifest_json,"
                " manifest_hash, classes_json, stats_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (version_id, dataset_id, _now(), json.dumps(manifest), manifest_hash,
                 json.dumps(classes), json.dumps(stats)))
        return self.get_dataset_version(version_id)

    def get_dataset_version(self, version_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM dataset_versions WHERE id = ?", (version_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["manifest"] = json.loads(d.pop("manifest_json"))
        d["classes"] = json.loads(d.pop("classes_json"))
        d["stats"] = json.loads(d.pop("stats_json"))
        return d

"""Run manager: FIFO queue, worker subprocess lifecycle, event fan-out.

Owns the single-writer path into the events table. Worker events flow:
mp.Queue → drainer thread → DB append (seq assigned) → asyncio broadcast to
WebSocket subscribers. Manager-originated transitions (queued, cancelled,
killed, interrupted, crash-detected failed) go through the same append path,
so the event log is complete regardless of who caused the transition
(DESIGN.md §5).
"""

import asyncio
import multiprocessing as mp
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty

from app.config import STREAM_SCHEMA_VERSION
from app.store.db import RESUMABLE_STATUSES, TERMINAL_STATUSES, Database
from app.training.worker import worker_main


def _env_stamp() -> dict:
    import subprocess

    import torch
    import torchvision

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=2,
                                cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        commit = None
    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "git_commit": commit}


@dataclass
class ActiveRun:
    run_id: str
    process: mp.Process
    queue: object
    stop_event: object
    kill_requested: bool = False
    stop_requested: bool = False
    drainer: threading.Thread = field(default=None, repr=False)


class RunManager:
    def __init__(self, db: Database, runs_root: Path):
        self.db = db
        self.runs_root = runs_root
        self.ctx = mp.get_context("spawn")
        self.loop: asyncio.AbstractEventLoop | None = None
        self.active: ActiveRun | None = None
        self.pending: deque[str] = deque()
        self.subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = threading.Lock()
        self._shutting_down = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def startup(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        # Recovery (§5): runs that were live when the previous process died.
        for run in self.db.runs_with_status(["running", "stopping"]):
            self._append(run["id"], "run_status",
                         {"status": "interrupted",
                          "error": "backend restarted while run was active"})
        for run in self.db.runs_with_status(["queued"]):
            self.pending.append(run["id"])
        self._try_start()

    def shutdown(self) -> None:
        self._shutting_down = True
        with self._lock:
            active = self.active
        if active and active.process.is_alive():
            active.process.terminate()

    # ── public API ────────────────────────────────────────────────────────

    def submit(self, config: dict, parent_run_id: str | None = None,
               dataset_version_id: str | None = None) -> dict:
        run_id = uuid.uuid4().hex[:12]
        config = {**config, "env": _env_stamp()}  # reproducibility (§4 run manifest)
        self.db.create_run(run_id, config, STREAM_SCHEMA_VERSION, parent_run_id,
                           dataset_version_id)
        self._append(run_id, "run_status", {"status": "queued"})
        with self._lock:
            self.pending.append(run_id)
        self._try_start()
        return self.db.get_run(run_id)

    def request_stop(self, run_id: str) -> None:
        """Graceful: worker finishes current batch, checkpoints, exits."""
        with self._lock:
            active = self.active
        if not active or active.run_id != run_id:
            raise LookupError("run is not currently running")
        if not active.stop_requested:  # double-stop is a no-op (§5)
            active.stop_requested = True
            self._append(run_id, "run_status", {"status": "stopping"})
            active.stop_event.set()

    def kill(self, run_id: str) -> str:
        """Immediate. On a queued run this is a cancel (§5)."""
        with self._lock:
            if run_id in self.pending:
                self.pending.remove(run_id)
                self._append(run_id, "run_status", {"status": "cancelled"})
                return "cancelled"
            active = self.active
        if not active or active.run_id != run_id:
            raise LookupError("run is not active")
        active.kill_requested = True
        active.process.terminate()
        return "killed"

    def resume(self, parent_run_id: str, overrides: dict | None = None) -> dict:
        parent = self.db.get_run(parent_run_id)
        if not parent:
            raise LookupError("no such run")
        if parent["status"] not in RESUMABLE_STATUSES:
            raise ValueError(
                f"run is {parent['status']}; resume is allowed from "
                f"{sorted(RESUMABLE_STATUSES)} (DESIGN.md §5)")
        ckpt = self.db.latest_checkpoint(parent_run_id)
        if not ckpt or not ckpt["path"]:
            raise ValueError("run has no checkpoint to resume from")
        config = {**parent["config"], **(overrides or {}),
                  "parent_run_id": parent_run_id, "resume_checkpoint": ckpt["path"]}
        return self.submit(config, parent_run_id=parent_run_id)

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.setdefault(run_id, set()).add(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        self.subscribers.get(run_id, set()).discard(q)

    # ── internals ─────────────────────────────────────────────────────────

    def _append(self, run_id: str, type_: str, payload: dict) -> None:
        if type_ == "run_status":
            # Terminal states are immutable (§5 state machine) — enforced at
            # the single-writer chokepoint so no race can violate it.
            current = self.db.get_run(run_id)["status"]
            if current in TERMINAL_STATUSES and payload["status"] != current:
                return
        event = self.db.append_event(run_id, type_, payload)
        if type_ == "run_status":
            self.db.set_run_status(run_id, payload["status"], payload.get("error"))
        elif type_ == "checkpoint_saved" and payload.get("path"):
            self.db.add_checkpoint(payload["checkpoint_id"], run_id, payload["epoch"],
                                   payload["global_step"], payload["path"],
                                   {"best_metric": payload.get("best_metric")})
        if self.loop:
            self.loop.call_soon_threadsafe(self._broadcast, run_id, event)

    def _broadcast(self, run_id: str, event: dict) -> None:
        for q in self.subscribers.get(run_id, set()):
            q.put_nowait(event)

    def _try_start(self) -> None:
        with self._lock:
            if self.active or self._shutting_down or not self.pending:
                return
            run_id = self.pending.popleft()
            run = self.db.get_run(run_id)
            run_dir = self.runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            queue = self.ctx.Queue()
            stop_event = self.ctx.Event()
            process = self.ctx.Process(
                target=worker_main,
                args=(run_id, run["config"], str(run_dir), queue, stop_event),
                daemon=True,
            )
            active = ActiveRun(run_id, process, queue, stop_event)
            self.active = active
        process.start()
        active.drainer = threading.Thread(target=self._drain, args=(active,), daemon=True)
        active.drainer.start()

    def _drain(self, active: ActiveRun) -> None:
        while True:
            try:
                msg = active.queue.get(timeout=0.5)
            except Empty:
                if not active.process.is_alive():
                    break
                continue
            self._append(active.run_id, msg["type"], msg["payload"])
        active.process.join()

        # Post-mortem: if the worker exited without a terminal run_status,
        # the manager records what happened (§5 failure model).
        status = self.db.get_run(active.run_id)["status"]
        if status not in TERMINAL_STATUSES:
            if active.kill_requested:
                self._append(active.run_id, "run_status", {"status": "killed"})
            elif self._shutting_down:
                pass  # next startup marks it interrupted
            else:
                self._append(active.run_id, "run_status",
                             {"status": "failed",
                              "error": f"worker exited unexpectedly "
                                       f"(exit code {active.process.exitcode})"})
        with self._lock:
            self.active = None
        self._try_start()

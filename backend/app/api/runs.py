"""REST API for runs (DESIGN.md §4 API surface)."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.security import issue_ticket, require_auth
from app.store.db import TERMINAL_STATUSES
from app.training import infer

router = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])

MAX_SERIES_POINTS = 500


class RunCreate(BaseModel):
    track: str = "mnist_mlp"
    epochs: int = 3
    lr: float = 0.1
    batch_size: int = 128
    hidden: list[int] = [128, 64]
    seed: int = 0
    device: str = "auto"
    parent_run_id: str | None = None
    # fake-worker knobs (tests / event-backbone development)
    batches_per_epoch: int | None = None
    delay: float | None = None


class RunAction(BaseModel):
    action: str  # "stop" | "kill"


class PredictBody(BaseModel):
    image: str  # base64 PNG (data URL accepted)


def manager(request: Request):
    return request.app.state.manager


def db(request: Request):
    return request.app.state.db


@router.get("/runs")
def list_runs(request: Request):
    return db(request).list_runs()


@router.post("/runs", status_code=201)
def create_run(body: RunCreate, request: Request):
    if body.parent_run_id:
        # Resume keeps the parent's config. Only fields the client explicitly
        # sent count as overrides — model defaults must not clobber the
        # parent's values (exclude_unset, not exclude_none).
        provided = body.model_dump(exclude_unset=True, exclude={"parent_run_id"})
        try:
            return manager(request).resume(body.parent_run_id,
                                           {"epochs": provided["epochs"]}
                                           if "epochs" in provided else {})
        except LookupError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))
    config = body.model_dump(exclude={"parent_run_id"}, exclude_none=True)
    return manager(request).submit(config)


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    run = db(request).get_run(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    run["latest_checkpoint"] = db(request).latest_checkpoint(run_id)
    return run


@router.get("/runs/{run_id}/events")
def get_events(run_id: str, request: Request, after_seq: int = 0):
    if not db(request).get_run(run_id):
        raise HTTPException(404, "no such run")
    return db(request).events_after(run_id, after_seq)


@router.get("/runs/{run_id}/series")
def get_series(run_id: str, request: Request):
    """Downsampled chart series (DESIGN.md §4): full fidelity stays in the DB."""
    if not db(request).get_run(run_id):
        raise HTTPException(404, "no such run")
    events = db(request).events_after(run_id, 0)
    batch = [e["payload"] for e in events if e["type"] == "batch_metrics"]
    epochs = [e["payload"] for e in events if e["type"] == "epoch_metrics"]
    if len(batch) > MAX_SERIES_POINTS:
        stride = -(-len(batch) // MAX_SERIES_POINTS)
        batch = batch[::stride]
    return {"batch": batch, "epochs": epochs}


@router.post("/runs/{run_id}/actions")
def run_action(run_id: str, body: RunAction, request: Request):
    run = db(request).get_run(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    if run["status"] in TERMINAL_STATUSES:
        raise HTTPException(409, f"run is already {run['status']} (terminal)")
    try:
        if body.action == "stop":
            manager(request).request_stop(run_id)
            return {"status": "stopping"}
        if body.action == "kill":
            return {"status": manager(request).kill(run_id)}
    except LookupError:
        raise HTTPException(409, f"run is {run['status']}; cannot {body.action}")
    raise HTTPException(400, "action must be 'stop' or 'kill'")


@router.post("/runs/{run_id}/predict")
def predict(run_id: str, body: PredictBody, request: Request):
    ckpt = db(request).latest_checkpoint(run_id)
    if not ckpt or not ckpt["path"]:
        raise HTTPException(409, "run has no checkpoint yet")
    try:
        return infer.predict(ckpt["path"], body.image)
    except infer.EmptyCanvasError:
        raise HTTPException(400, "canvas is empty — draw a digit first")


@router.post("/ws-ticket")
def ws_ticket():
    """Short-lived single-use ticket for WebSocket auth (DESIGN.md §3)."""
    return {"ticket": issue_ticket()}

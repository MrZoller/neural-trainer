"""neural-trainer backend — FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets as datasets_api
from app.api import runs as runs_api
from app.api import ws as ws_api
from app.config import DATA_DIR
from app.device import device_info
from app.store.db import Database
from app.training.manager import RunManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database(DATA_DIR / "neural-trainer.db")
    manager = RunManager(db, DATA_DIR / "runs")
    manager.startup(asyncio.get_running_loop())
    app.state.db = db
    app.state.manager = manager
    yield
    manager.shutdown()


app = FastAPI(title="neural-trainer", lifespan=lifespan)

# Dev frontend origins only; tightened when remote mode lands (DESIGN.md §3,
# "Security posture").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs_api.router)
app.include_router(datasets_api.router)
app.include_router(ws_api.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/device")
def device() -> dict:
    return device_info()

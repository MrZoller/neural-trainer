"""Paths and environment configuration."""

import os
from pathlib import Path

# Repo-level data/ directory (DESIGN.md §3 Storage); NT_DATA_DIR overrides for tests.
DATA_DIR = Path(os.environ.get("NT_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))

# Remote-mode shared token (DESIGN.md §3 Security posture). Unset = localhost dev,
# no auth enforced.
TOKEN = os.environ.get("NT_TOKEN")

STREAM_SCHEMA_VERSION = 1

#!/usr/bin/env bash
# Start backend (uvicorn :8000) and frontend (vite :5173) dev servers together.
# Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

trap 'kill 0' EXIT

# --timeout-graceful-shutdown: open WebSockets otherwise block reloads forever
(cd backend && uv run uvicorn app.main:app --reload --port 8000 --timeout-graceful-shutdown 3) &
(cd frontend && npm run dev) &
wait

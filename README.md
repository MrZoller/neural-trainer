# neural-trainer

Train a real neural net. See what it's actually doing.

A guided app for training PyTorch models end-to-end: pick a task, curate data, watch
training with honest under-the-hood explanations, then use the trained model. The
sequel to [neural-viz](https://github.com/MrZoller/neural-viz).

**Design:** see [DESIGN.md](DESIGN.md). **Current state:** Phase 0 (scaffolding).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (backend Python/deps management)
- Node.js 18+ / npm 9+

## Development

```bash
./scripts/dev.sh
```

or run the pieces separately:

```bash
# Backend — FastAPI + PyTorch on http://127.0.0.1:8000
cd backend && uv run uvicorn app.main:app --reload

# Frontend — Vite dev server on http://localhost:5173 (proxies /api and /ws to :8000)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 — the home screen shows the backend connection and which
compute device PyTorch selected (CUDA / MPS / CPU).

## Layout

```
backend/    FastAPI + PyTorch server (uv project)
  app/
    api/        REST + WS routes            (Phase 1)
    training/   loops, models, hooks         (Phase 1)
    datasets/   import, curation             (Phase 2A)
    store/      SQLite + files               (Phase 1)
frontend/   React 18 + Vite 5 + Tailwind 3 (same stack as neural-viz)
scripts/    dev helpers
data/       app state — datasets, runs, DB (gitignored; copying it migrates everything)
```

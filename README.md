# neural-trainer

[![CI](https://github.com/MrZoller/neural-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/MrZoller/neural-trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Train a real neural net. See what it's actually doing.

A guided app for training PyTorch models end-to-end: pick a task, curate data, watch
training with honest under-the-hood explanations, then use the trained model. The
sequel to [neural-viz](https://github.com/MrZoller/neural-viz).

**Design:** see [DESIGN.md](DESIGN.md). **Current state:** Phase 2A — MNIST from
scratch (Track 1) and custom image classification via transfer learning (Track 2:
folder import with EXIF-burst capture groups, labeling grid, leakage-safe splits,
dataset versioning, fine-tuned MobileNetV3, upload-image inference).

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

## Tests

```bash
cd backend && uv run pytest       # backend test suite
cd backend && uv run ruff check . # lint
cd frontend && npm run build      # frontend production build
```

CI runs the same commands on every pull request.

## Contributing

Bug reports, fixes, and documentation improvements are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and conventions, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

To report a security problem, follow [SECURITY.md](SECURITY.md) rather than
opening a public issue. Note the intended deployment boundary: this backend is
built to run on a machine you control, reached over a private network. Exposing
it to the open internet is out of scope.

## License

[MIT](LICENSE) © Chris Zoller

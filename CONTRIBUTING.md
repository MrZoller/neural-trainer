# Contributing to neural-trainer

Thanks for your interest. This is a personal learning project first, so the bar for
new features is "does it make what the model is doing more visible?" — but bug
reports, fixes, and documentation improvements are all welcome.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- Read [DESIGN.md](DESIGN.md). It is the source of truth for architecture, the
  two training tracks, the event-sourced run model, and the storage layout.
  Sections are referenced from code comments (`DESIGN.md §4`) — keep those
  references accurate when you change behaviour.
- For anything larger than a bug fix, open an issue first so we can agree on the
  approach before you write code. The non-goals in DESIGN.md §1 are deliberate.

## Development setup

Prerequisites: [uv](https://docs.astral.sh/uv/) and Node.js 18+ / npm 9+.

```bash
git clone https://github.com/MrZoller/neural-trainer.git
cd neural-trainer
./scripts/dev.sh          # backend on :8000, frontend on :5173
```

`uv` creates and syncs `backend/.venv` on first `uv run`; there is no separate
install step. The frontend needs `npm install` once (`scripts/dev.sh` assumes it
has been run).

## Checks

Run these before opening a pull request — CI runs the same commands.

```bash
# Backend
cd backend
uv run pytest              # test suite
uv run ruff check .        # lint

# Frontend
cd frontend
npm run build              # production build must succeed
npm run format:check       # Prettier
```

`uv run ruff format` and `npm run format` will fix most style findings in place.
Note that `ruff format` is **not** enforced in CI yet — the existing backend code
predates it, so running it repo-wide would produce a large unrelated diff. Format
only the code you touch.

## Conventions

- **Python:** type hints on public functions; ruff for lint; 100-column lines.
  Module docstrings say what the module is for and cite the DESIGN.md section.
- **JS/React:** function components with hooks, Tailwind utility classes, no
  semicolons, single quotes (see `frontend/.prettierrc`).
- **Tests:** `backend/tests/`, pytest, one file per concern
  (`test_events.py`, `test_lifecycle.py`, `test_datasets.py`). New behaviour in
  the store, run lifecycle, or dataset pipeline needs a test.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) —
  `type(scope): summary`, where type is one of `feat`, `fix`, `refactor`, `test`,
  `docs`, `chore`.

## Pull requests

1. Branch off `main`.
2. Keep the change focused; unrelated reformatting makes review harder.
3. Fill in the pull request template — especially how you verified the change.
   "Trained a 3-epoch MNIST run and watched the loss curve" is a real answer.
4. CI must be green before merge.

## Reporting bugs

Use the bug report template and include your platform and compute device (the
home screen shows whether PyTorch selected CUDA, MPS, or CPU) — a surprising
number of training bugs are device-specific.

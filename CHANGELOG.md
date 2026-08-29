# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## Phase 2A — 2026-07-03

Custom image classification via transfer learning (Track 2).

### Added

- Folder import with EXIF-burst capture groups, so near-duplicate photos stay
  together instead of being split across train and validation.
- Labeling grid for curating imported images.
- Leakage-safe train/validation splits that respect capture groups.
- Dataset versioning: a run freezes its dataset into an immutable manifest, so
  later edits to the dataset can't change what a finished run trained on.
- Fine-tuning of a pretrained MobileNetV3 backbone with a new classifier head.
- Upload-an-image inference against a trained custom model.

## Phase 0 + 1 — 2026-07-03

Scaffolding and the MNIST walking skeleton (Track 1).

### Added

- FastAPI + PyTorch backend as a `uv` project; React 18 + Vite 5 + Tailwind 3
  frontend.
- Event-sourced training backbone: append-only event store with per-run
  monotonic sequence numbers, replayed to reconstruct run state.
- Training runs in a worker subprocess with checkpointing, stop/kill lifecycle,
  and resume from checkpoint.
- Live run streaming over WebSocket, with charts for loss and accuracy.
- MNIST training from scratch on a small MLP, plus draw-a-digit inference.
- Automatic compute device selection (CUDA / MPS / CPU).
- Remote-mode auth: optional `NT_TOKEN` bearer token for REST and single-use
  tickets for WebSocket connections.

[Unreleased]: https://github.com/MrZoller/neural-trainer/compare/main...HEAD

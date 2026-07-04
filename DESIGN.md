# neural-trainer — Design Document

**Status:** v3, build-ready · 2026-07-03 (two rounds of external review incorporated)

A guided app for training real neural networks with PyTorch: pick a task, curate data,
watch training with honest under-the-hood explanations, then use the trained model.
The spiritual sequel to [neural-viz](https://github.com/MrZoller/neural-viz) — same
"make the math visible" ethos, applied to real models on real data.

---

## 1. Goals

1. **Learn by doing.** The user trains an actual model end-to-end and understands what
   happened at each step — not a black-box AutoML experience.
2. **Honest explanations.** Every visualization is computed from real values (metrics,
   gradients, activations). Diagnostics are framed as evidence-backed *possibilities*,
   not verdicts. Simplifications are labeled, as in neural-viz.
3. **Personally useful output.** After training, the user can point a webcam or drop an
   image and get predictions from *their* model.
4. **Runs on Chris's hardware.** MacBook / Mac mini (Apple Silicon → MPS) and Legion PC
   (Ubuntu + NVIDIA GPU → CUDA), with a path to cloud GPUs later that requires no
   re-architecture.

### Non-goals for v1

- Object detection / segmentation (classification only)
- Web-scraped training data (user-provided + standard datasets only)
- Multi-user / auth beyond a single shared token for private-network remote access
- Distributed or multi-GPU training
- Text / audio domains (image classification first; architecture shouldn't preclude them)
- True in-process pause/resume — the lifecycle model is checkpoint-based (§5)

---

## 2. The two training tracks

| | Track 1 — From Scratch | Track 2 — Custom Task |
|---|---|---|
| **Purpose** | Maximum learning value | A model that's actually yours and actually works |
| **Data** | MNIST, CIFAR-10 (via torchvision) | User-curated: webcam captures, imported folders |
| **Models** | Small MLP and small CNN, defined in readable code | Pretrained backbone (MobileNetV3 / ResNet18), frozen features + new classifier head |
| **Why it works** | 50k+ labeled examples exist | Transfer learning needs only ~30–100 images/class |
| **Wizard depth** | Deep: architecture choices, init, optimizer, every knob explained | Focused: what "frozen" means, why 50 photos suffice, when to unfreeze deeper layers |

The app explicitly teaches *why* Track 2 exists: from-scratch training on 50 photos
fails, and the wizard demonstrates rather than hides this (an optional "try it from
scratch anyway" run makes the point empirically).

---

## 3. System architecture

```
┌─ Browser (any machine) ───────────┐        ┌─ Backend (machine with compute) ──┐
│  React 18 + Vite + Tailwind       │  REST  │  FastAPI + uvicorn                │
│  · wizard & narration UI          │◄──────►│  · dataset store (SQLite + files) │
│  · live charts (loss, layers)     │   WS   │  · training worker (subprocess)   │
│  · dataset studio                 │◄───────│  · PyTorch: CUDA / MPS / CPU      │
│  · inference playground           │        │  · checkpoints & run history      │
└───────────────────────────────────┘        └───────────────────────────────────┘
```

Key properties:

- **The backend is location-independent.** Same code on the Mac mini, the Legion, or a
  cloud GPU box. Remote use = explicit remote flag + the UI pointed at that host.
  (Tailscale makes the Mac↔Legion link trivial and secure; recommended in docs.)
- **Training runs in a worker subprocess**, not in the API process. The API stays
  responsive; a crash in training can't take down the server. Worker → API
  communication via a multiprocessing queue; API → browser via WebSocket. One
  concurrent run in v1; additional run requests queue FIFO.
- **Device selection is automatic:** `cuda` if available, else `mps`, else `cpu` — with
  an override in run config and the active device always visible in the UI.

### Security posture (defaults from Phase 1, not retrofitted in Phase 5)

Training endpoints can burn GPU hours, fill disks, and hold personal webcam images —
they are not harmless.

- Bind `127.0.0.1` by default; non-localhost bind requires an explicit `--remote` flag
- Non-localhost mode requires a shared token (env var); UI shows a persistent
  "remote mode" banner
- Upload size limits; image MIME allowlist (jpeg/png/webp); CORS origin allowlist
- Tokens never ride in WebSocket query strings (they leak into logs and browser
  history): REST uses `Authorization: Bearer`; the WS connects with a short-lived
  ticket fetched over authenticated REST
- Supported deployment boundary: private network (Tailscale). Exposing the backend to
  the open internet is explicitly out of scope.

### Storage

```
data/
├── neural-trainer.db        # SQLite: projects, datasets, dataset_versions, images,
│                            #   labels, runs, events, metrics, embeddings
├── datasets/<dataset_id>/   # original images (content-addressed filenames)
├── runs/<run_id>/           # checkpoints, run manifest, exported metrics
└── cache/                   # torchvision downloads
```

SQLite + files keeps everything portable — copying `data/` migrates the whole app
state between machines.

### Dataset versioning

Two distinct concepts:

- **Dataset** (living) — the user adds, relabels, and excludes images freely.
- **Dataset version** (frozen) — starting a run snapshots a manifest:
  `[(image content-hash, label, split assignment)]`, itself content-hashed, stored in
  the DB and in `runs/<id>/manifest.json`.

Runs reference a `dataset_version_id`, so run history and comparisons stay meaningful
after the living dataset changes. Because image files are content-addressed, a
manifest is just a list of pointers — versions are nearly free.

**Deletion policy:** an image referenced by any frozen dataset version cannot be
hard-deleted. The UI offers *exclude from future versions* instead; hard delete is
possible only after deleting the dependent runs. Old runs stay reproducible — history
never silently breaks.

---

## 4. Backend design

**Stack:** Python 3.11+, uv, FastAPI, PyTorch + torchvision, Pillow, SQLite (via
sqlite3 or SQLModel — decide in Phase 0).

### Training events — treated as a real internal API

Everything the UI shows is derived from a persisted, typed event stream. Envelope:

```
event_id · run_id · seq · timestamp · type · payload
```

- `seq` is a per-run monotonic sequence number. WebSocket clients reconnect with
  `last_seq` and the server replays the gap — this one field buys refresh-safety,
  missed-message recovery, and replayable history.
- Each run records a single `stream_schema_version`; payload shapes may only change
  with a version bump.
- **Full fidelity is stored; charts get downsampled series.** `batch_metrics` events
  are throttled at emission, and chart endpoints (`GET /api/runs/{id}/series?...`)
  downsample server-side to ~500 points so long runs don't melt the browser, while the
  DB keeps everything for debugging.
- **Storage:** `UNIQUE(run_id, seq)` with an index on `(run_id, seq)`; SQLite in WAL
  mode; a single event-writer path with batched inserts. A replay-compatibility test
  feeds saved event logs from prior `stream_schema_version`s through the current
  frontend parser.

| Event type | Payload | Cadence |
|---|---|---|
| `run_status` | queued / running / stopping / stopped / done / failed / interrupted | on change |
| `batch_metrics` | loss, lr, throughput | every N batches (throttled) |
| `epoch_metrics` | train/val loss & accuracy, per-class recall | per epoch |
| `layer_stats` | per-layer grad norm, weight norm, % dead ReLUs | per epoch |
| `sample_predictions` | fixed probe images + current predictions | per epoch |
| `diagnostic` | rule id + evidence values (see §7) | when a rule fires |
| `checkpoint_saved` | path, epoch, val metrics | per save |

### API surface (sketch)

```
REST
  GET/POST/DELETE  /api/datasets                  # CRUD (living datasets)
  POST             /api/datasets/{id}/images      # upload (multipart, batch)
  PATCH            /api/images/{id}               # relabel, exclude
  GET              /api/datasets/{id}/report      # quality report (§8)
  POST             /api/runs                      # start run (freezes dataset version;
                                                  #   parent_run_id + checkpoint = resume)
  GET              /api/runs/{id}                 # status, manifest, persisted metrics
  GET              /api/runs/{id}/series          # downsampled chart series
  POST             /api/runs/{id}/actions         # stop (graceful) | kill (immediate)
  POST             /api/runs/{id}/predict         # inference: image → probs (+ saliency)
  GET              /api/device                    # active compute device info

WS
  /ws/runs/{id}?last_seq=N                        # live event stream with gap replay
```

### Run manifest — reproducibility as a visible feature

Every run stores: model architecture + pretrained-weights version, `dataset_version_id`,
train/val split, random seed, transforms/augmentations, optimizer config, device,
PyTorch/torchvision versions, and git commit if available. The Runs screen's
"compare experiments" view diffs manifests — comparison is only meaningful when you
can see exactly what changed.

### Models

- **Track 1:** `SimpleMLP` and `SmallCNN` written as plain, heavily-commented
  `nn.Module` classes — the code itself is a teaching artifact, mirroring how
  neural-viz keeps all math in one readable file.
- **Track 2:** torchvision backbone (default MobileNetV3-Small for speed; ResNet18 as
  the "classic" option) with `requires_grad=False` on features, new `nn.Linear` head.
  Advanced option: unfreeze the last block ("fine-tune deeper") with a lower LR.

---

## 5. Run lifecycle & failure model

The action model is deliberately checkpoint-based — no attempt to freeze a live
training loop in memory:

- **stop** (graceful) — worker finishes the current batch, saves a checkpoint, exits;
  run marked `stopped`.
- **kill** — immediate termination; the last durable checkpoint stands.
- **resume** — a new run created from a checkpoint with `parent_run_id` lineage; the
  UI stitches parent+child charts and presents the pair as pause/resume.

### Run state machine

```
queued    → running | cancelled
running   → stopping | killed | done | failed | interrupted
stopping  → stopped | killed | failed
```

- **Terminal states** — `cancelled`, `stopped`, `killed`, `done`, `failed`,
  `interrupted` — are immutable once entered.
- **"Resumed" is deliberately not a state.** A terminal run that has been resumed
  keeps its terminal state; resume lineage is derived from the child's
  `parent_run_id`. (A separate `resumed_by_child` state would make terminal states
  mutable and every consumer handle one more transition.)
- **Resume** is allowed from `stopped` / `killed` / `failed` / `interrupted` iff a
  checkpoint exists; it creates a child run in `queued`. Resume without a checkpoint
  is rejected with an explanation.
- **Idempotency:** stop while `stopping` is a no-op; kill while `stopping` escalates
  to `killed`; kill on a `queued` run → `cancelled` (never ran, nothing to
  checkpoint); any action on a terminal run is a no-op error.

### Checkpoint contents (defined before Phase 1 code is written)

A checkpoint is complete only if resuming reproduces *training behavior*, not merely
weights:

```
model_state_dict · optimizer_state_dict · scheduler_state_dict (if any)
epoch · global_step · best-metric-so-far
RNG states (Python, torch CPU, CUDA/MPS)
class mapping · transform/normalization config
run-config snapshot · parent_run_id / checkpoint_id lineage
```

Loading weights alone is inference export — a different operation, not resume.
Cross-device caveat (recorded in the run manifest): resuming on a different device
(CUDA↔MPS) is permitted but not bit-identical — RNG streams differ across backends;
each run segment records its device.

Named failure cases and v1 behavior:

| Case | Behavior |
|---|---|
| Browser refresh / WS disconnect | Events are persisted; client reconnects with `last_seq`, server replays the gap |
| Worker crash | API detects exit, marks run `failed` with captured stderr tail; last checkpoint remains resumable |
| Backend restart | Runs that were `running` are marked `interrupted` on startup; resumable from last checkpoint |
| GPU OOM | Caught in worker → run `failed` + diagnostic card (evidence: batch size, image size, device memory; try next: smaller batch) |
| Disk full / checkpoint save fails | Fail loudly, never continue silently; pre-run free-space check |
| Second run started | Queues FIFO (single worker in v1); queue position visible |

---

## 6. Frontend design

**Stack:** React 18 + Vite + Tailwind (+ Recharts) — same as neural-viz, so styling
sensibilities and chart patterns carry over directly.

### Screen inventory

1. **Home** — project list, backend connection status (host + device badge:
   "Connected to legion · CUDA · RTX ____"), remote-mode banner when applicable.
2. **New-project wizard** — choose track; Track 1: pick dataset + architecture with
   explanations; Track 2: name your classes.
3. **Dataset studio** (Track 2) — drag-drop folder import, labeling grid, class-balance
   bar, group-aware train/val split view (§8), quality report panel; webcam
   burst-capture and augmentation preview arrive in Phase 2B.
4. **Train** — the centerpiece, built with **progressive disclosure** (all views feed
   from the same event stream; nothing is mocked):
   - **Watch** (default): train-vs-val loss/accuracy charts, probe-image strip (fixed
     samples — validation images, clearly marked — re-predicted each epoch so you *see*
     learning happen), narration feed of diagnostic cards
   - **Inspect**: block-level network diagram with per-layer gradient-norm and
     dead-ReLU coloring, weight norms, LR schedule
   - **Debug**: raw metrics, throughput, event log, run manifest
   - Config panel (pre-run): every knob (LR, batch size, epochs, optimizer,
     augmentation) with a one-paragraph plain-language explanation
5. **Evaluate** — confusion matrix, per-class precision/recall, "worst predictions"
   gallery, saliency overlay: *"highlights the image regions that most influenced this
   prediction"* (Grad-CAM; deliberately not phrased as "the model looked here").
6. **Playground** — webcam live-prediction or drag-drop, confidence bars, saliency
   toggle. Works against any completed run.
7. **Runs** — history table with lineage (resume chains), manifest diff between runs,
   overlaid loss curves for comparing experiments.

---

## 7. Explanation layer (the differentiator)

Pre-authored narration cards keyed to **diagnostic rules** — simple detectors running
server-side over the event stream. Because the rules are heuristics, every card uses a
hedged, evidence-first template. This *is* the honesty ethos: real evidence, candid
uncertainty.

```
Possible issue · <name>
Evidence          the actual numbers/series that fired the rule
Likely reading    the most common interpretation
Also possible     alternative explanations (small val set, bad split, mislabels, …)
Try next          concrete knobs
Go deeper         link to the full story
```

| Rule | Detector (v1 heuristic) | Card teaches |
|---|---|---|
| Overfitting | val loss ↑ over k evals while train loss ↓ | memorization vs. generalization; also possible: tiny/unlucky val split, mislabeled val images |
| Train/val leakage | near-duplicates straddle the split (§8) | why validation scores can lie; leakage-safe splitting |
| LR too high | loss oscillation / divergence early in run | step size vs. loss-surface curvature |
| Plateau | relative improvement < ε over window | LR schedules; when a model is "done" |
| Underfitting | train accuracy low after warm-up epochs | capacity; feature quality; task difficulty |
| Dead ReLUs | > x% units zero across an epoch (per layer) | why neurons die; init & LR connections (ties back to neural-viz's dead-ReLU callouts) |
| Vanishing/exploding grads | per-layer grad-norm ratio across depth | why depth is hard; normalization, residuals |
| Class imbalance bias | per-class recall spread > threshold | why accuracy alone lies; balanced data / weighted loss |
| GPU OOM | worker exception | memory vs. batch size / image size |

Card content lives in a YAML/JSON content file, versioned with the code — writing
these well is authoring work, not engineering risk, and it's where most of the
project's teaching value concentrates.

---

## 8. Dataset quality tooling

### Leakage-safe splitting (first-class, from Phase 2A — not deferred to Phase 4)

Near-duplicates on both sides of a train/val split make validation scores lie, and
webcam burst capture makes near-duplicate frames the *default* case, not an edge case.

- Every image carries a **capture group**, and split assignment operates on groups,
  never individual images. Group assignment per source:
  - **Webcam capture** (2B): one burst session = one group.
  - **Folder import** (2A): groups inferred from EXIF timestamps — photos in the same
    folder taken within a short window (~10s) form one group; per-file fallback when
    EXIF is absent. Groups are visible and editable in the labeling grid.
  - Deliberately rejected: "one import operation = one group." It's simple but wrong
    in the other direction — importing a whole class folder at once would fuse the
    class into a single unsplittable group, leaving validation empty for that class.
- If a class's images collapse into very few groups, the studio warns that validation
  will be optimistic ("all 48 dog photos come from one session").
- Phase 4 upgrades grouping: embedding near-duplicate clusters are merged into capture
  groups before splitting, and a leakage diagnostic fires if near-duplicates are found
  straddling an existing split.

### Embedding-based tooling (Phase 4)

Built on backbone embeddings (each image run once through the frozen Track 2 backbone;
vectors cached in SQLite):

- **Near-duplicate detection** — cosine-similarity pairs above threshold, surfaced as
  merge/remove suggestions. Pairwise is fine at hand-curated scale (≤ ~10k images).
- **Mislabel suggestions** — images where a cross-validated model confidently disagrees
  with the assigned label, ranked by confidence. Presented as "double-check these,"
  never auto-relabeled.

### Quality report — components, no composite score

No headline "Health: 72/100" — composite scores invite gaming and explain nothing.
Instead, a component checklist, each line linking to its explanation card (§7):

```
Class balance          good
Images per class       weak   (cat: 21 — aim for 30+)
Duplicate risk         high   (3 burst clusters look near-identical)
Variety within class   low    (dog photos: 1 location, 1 lighting)
Leakage risk           high   (near-duplicates straddle your split)
```

---

## 9. Phase plan

Each phase ends with something usable. Revised after review: the run/event/
dataset-version foundation is deliberately front-loaded (Phase 1), the custom track is
split so webcam UX doesn't block the useful loop, and leakage-safe splitting ships
with the first custom-track milestone.

| Phase | Deliverable | Proves |
|---|---|---|
| **0** | Repo scaffolding: `backend/` (uv + FastAPI skeleton), `frontend/` (Vite app), dev scripts, this doc | dev loop works on Mac |
| **1 — Walking skeleton + lifecycle backbone** | Train MNIST MLP from scratch; persisted event stream with `seq` replay across refresh; stop / kill / resume-from-checkpoint; localhost-default security; draw-a-digit inference | full pipe **and** the run/event model that everything else sits on |
| **2A — Custom track core** | Session-aware folder import (EXIF burst grouping, §8) → labeling grid → group-aware split → fine-tune backbone head → upload-image prediction; dataset versioning + run manifests | the personally-useful loop, minus webcam |
| **2B — Webcam** | Burst capture (with capture groups), live webcam playground, augmentation preview | webcam UX isolated where its browser-permission/HTTPS quirks can't block 2A |
| **3 — Explanation layer** | Diagnostic rules + hedged narration cards; Inspect view (layer stats); probe strip; Grad-CAM; Evaluate screen | the teaching value |
| **4 — Curation quality** | Embedding cache; near-dup clusters folded into split groups; leakage detector; mislabel suggestions; component quality report | dataset tooling |
| **5 — Remote & cloud** | Legion headless setup docs (systemd unit, Tailscale); remote-mode polish; cloud-GPU deploy recipe | train on Legion / cloud, drive from MacBook |

**Phase 1 build order** (inside-out, so event/UI debugging is decoupled from PyTorch
debugging): DB schema + event append/replay API → **fake worker emitting synthetic
events** → WS reconnect with `last_seq` → real MNIST worker → checkpoint save +
resume-as-child-run → stop/kill lifecycle → Watch UI charts from persisted events →
draw-a-digit inference → security defaults.

Verification per phase: Phase 1 MNIST MLP should reach ~97% test accuracy in under a
minute on any of the three devices — **and the draw-a-digit canvas must work credibly
on hand-drawn input**, which is a preprocessing problem, not an accuracy problem:
canvas strokes go through the MNIST-style path (grayscale → invert → crop to content →
center by mass → resize to 28×28 with padding → training normalization), and the UI
shows the preprocessed 28×28 image the model actually sees alongside the prediction —
a debugging aid and the app's first distribution-shift lesson in one. Phase 2A/2B
should hit >90% on a 3-class, ~50-images-per-class dataset **evaluated against a
held-out capture session (different day/lighting)** — the target must not be reachable
via leakage, or the app's first real demo would exhibit exactly the failure mode §8
exists to prevent.

---

## 10. Risks & mitigations

- **State complexity is the real long-term risk.** Datasets, versions, runs, events,
  checkpoints, queues, and browser sessions interact; the mitigation is making the
  run/event/dataset-version model boring and robust *first* (Phase 1) and deriving all
  UI from it, before any flashy teaching features land.
- **Webcam access requires HTTPS or localhost** in browsers. Fine in dev; for remote
  use, Tailscale HTTPS certs or a localhost-forwarded port solve it. Isolated in
  Phase 2B / 5.
- **MPS op gaps** — occasional torchvision ops lack MPS kernels. Mitigation: CPU
  fallback flag, and CI-style smoke test of both tracks on MPS early (Phase 1–2).
- **Narration quality** is the project's real bar — schedule writing time for cards,
  not just code (Phase 3 is deliberately its own phase). The hedged template keeps
  cards honest even when heuristics misfire.
- **Scope gravity** toward detection/segmentation/more domains — non-goals list exists
  for a reason; the event-stream + track abstractions keep the door open without
  building any of it now.

## 11. Open questions (deferrable)

- SQLModel vs. raw SQLite for the store (decide in Phase 0; low stakes at this scale)
- MobileNetV3 vs. ResNet18 as the *default* backbone (benchmark both in Phase 2A)
- Whether Track 1 gets a CIFAR-10 CNN in Phase 1 or Phase 3 (leaning Phase 3, keeping
  Phase 1 minimal)
- Chart downsampling algorithm (simple stride vs. LTTB — decide when charts exist)

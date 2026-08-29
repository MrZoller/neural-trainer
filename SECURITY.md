# Security Policy

## Supported versions

neural-trainer is pre-1.0 and ships no releases yet. Only the current `main`
branch is supported — fixes land there.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it through GitHub's private vulnerability reporting:
[Security → Report a vulnerability](https://github.com/MrZoller/neural-trainer/security/advisories/new).

Include what you can: affected endpoint or component, reproduction steps, and
the impact you believe it has. This is a hobby project maintained by one person
— expect an initial response within about a week, not within hours.

## Intended deployment boundary

This matters for judging whether something is a vulnerability.

neural-trainer is designed to run on a machine you control, reached from your own
devices over a **private network** (for example Tailscale). Exposing the backend
to the open internet is explicitly out of scope (DESIGN.md §3). Reports whose
only impact requires an internet-exposed instance will be documented rather than
patched.

### What is enforced today

- REST authenticates with `Authorization: Bearer <token>`, compared using
  `secrets.compare_digest` — **only when `NT_TOKEN` is set**. See the gaps below.
- WebSockets authenticate with a short-lived, single-use ticket fetched over
  authenticated REST. Tokens never travel in WebSocket query strings, where they
  would leak into logs and browser history.
- Dataset uploads are restricted to an explicit content-type allowlist
  (jpeg/png/webp) and must decode as images; anything else is rejected.
- Uploads over 10 MB are rejected.

### Known gaps — operator responsibility

These are documented rather than fixed. They are **known**, so reporting them
again is not necessary; reporting a way to bypass something in the list above
is.

- **Nothing enforces the bind address / token pairing.** DESIGN.md §3 describes
  non-localhost binding as requiring `NT_TOKEN`, but that check does not exist
  yet: there is no startup or bind-address validation, and `require_auth`
  returns immediately when `NT_TOKEN` is unset. Binding to `0.0.0.0` without a
  token therefore starts a fully unauthenticated server. **Set `NT_TOKEN`
  yourself before binding anywhere but `127.0.0.1`.** `scripts/dev.sh` and the
  documented `uvicorn` invocations use uvicorn's `127.0.0.1` default, but that
  is a default, not a guard.
- **Upload limits apply after buffering, not before.** `upload_images` reads
  each file fully into memory before the 10 MB check, and `PredictBody.image` is
  an unconstrained base64 string that is decoded whole. Both bound what is
  *stored*, not what is *allocated*, so a large request can still spike process
  memory. Put a body-size limit in front of the app if that matters to you.

Training endpoints can consume GPU time, fill disks, and read personal images.
Treat unauthenticated access to them as a real finding — but note that within
the gaps above, "unauthenticated" is the documented behaviour when `NT_TOKEN` is
unset, not a vulnerability.

Defeating anything in **What is enforced today** — bypassing bearer auth with a
token set, replaying or forging a WebSocket ticket, getting a non-image past the
allowlist, path traversal in the dataset or checkpoint store, or arbitrary code
execution via an uploaded file or checkpoint — is in scope and worth reporting.

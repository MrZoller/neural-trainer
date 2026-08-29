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

Within that boundary, the design holds these to be security-relevant:

- The backend binds `127.0.0.1` by default. Binding elsewhere requires the
  `NT_TOKEN` shared token; with no token set, no auth is enforced.
- REST authenticates with `Authorization: Bearer <token>`, compared with
  `secrets.compare_digest`.
- WebSockets authenticate with a short-lived, single-use ticket fetched over
  authenticated REST. Tokens must never travel in WebSocket query strings, where
  they leak into logs and browser history.
- Uploads are size-limited and restricted to decodable images.
- Training endpoints can consume GPU time, fill disks, and read personal images.
  Treat unauthenticated access to them as a real finding.

Bypassing any of the above — auth bypass, ticket replay, path traversal in the
dataset or checkpoint store, or arbitrary code execution via an uploaded file or
checkpoint — is in scope and worth reporting.

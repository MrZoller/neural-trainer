## What

<!-- What does this change, and why? Link the issue if there is one: Fixes #123 -->

## How it was verified

<!--
Beyond CI. For anything touching training, say what you actually ran —
"3-epoch MNIST run on MPS, loss curve looked normal, resumed from checkpoint".
-->

- [ ] `cd backend && uv run pytest`
- [ ] `cd backend && uv run ruff check .`
- [ ] `cd frontend && npm run format:check && npm run build`
- [ ] Exercised the change in the running app

## Notes for the reviewer

<!-- Anything non-obvious: trade-offs, follow-ups, things you're unsure about. -->

---

- [ ] DESIGN.md updated if this changes architecture or behaviour it describes
- [ ] CHANGELOG.md updated under `## [Unreleased]` if this is user-visible

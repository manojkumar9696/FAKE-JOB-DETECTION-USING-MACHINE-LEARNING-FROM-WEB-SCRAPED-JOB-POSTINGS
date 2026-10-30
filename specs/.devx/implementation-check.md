# Implementation Check

> Local readiness gate for IDE code generation.

## Status

CONCERNS

## Required Before Coding

- [ ] Run `bash specs/.devx/discover-workspace.sh` in the target IDE workspace.
- [ ] Review `specs/.devx/project-context.md`.
- [ ] Review `specs/.devx/current-state.md`.
- [ ] Review the selected feature change map under `specs/.devx/change-maps/`.
- [ ] Review local Golden Repo guidance under `specs/.devx/guidance/` when present.
- [ ] Confirm the owning code area for the feature.

## Architecture Mode

- monolith

## Notes

Do not start implementation from a Golden Repo link or Astra-only context. The IDE agent must use local files committed or copied with the specs.

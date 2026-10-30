# Project Context

> Portable implementation context for AI-assisted code generation.
> Run `bash specs/.devx/discover-workspace.sh` in the target IDE workspace to refresh this file from actual code.

## Project

- Name: DemoTest
- Architecture mode: monolith
- Context status: Pending workspace discovery

## Technology Stack And Versions

- Pending: discover from package files, project files, lock files, and repository manifests.

## Code Organization Patterns

- Inspect existing folders before adding new ones.
- Prefer extending existing modules, components, services, and tests over creating parallel implementations.
- Keep generated specs under `specs/`; implementation should happen in the owning application/service code.

## Naming And Style Rules

- Match existing naming conventions in the target repo.
- Reuse established import aliases, formatter settings, lint rules, and test conventions.

## Testing Approach

- Inspect nearby tests before adding new test structure.
- Reuse existing unit, integration, API, UI, and e2e test tools where present.

## Critical Implementation Rules

- Read `specs/.devx/current-state.md`, `specs/.devx/implementation-check.md`, and the feature change map before coding.
- Read local Golden Repo guidance files when present under `specs/.devx/guidance/`.
- Existing code patterns override generic guidance when they conflict.
- Do not create a new repo, service, route family, or UI pattern unless the spec explicitly requires it.

## Feature Presence Rule

- A tracked feature is actionable only if its `specs/<slug>/specs.md` exists in this checkout. Features are pushed selectively, so a tracked-but-absent folder is EXPECTED and is not an error.
- If a PENDING feature's `specs/<slug>/specs.md` is missing, SKIP it: report the skipped slug, move on to the next PENDING feature, and never modify or remove its `features.json`/`tracker.json` entry.
- Never set a feature to `done`/`COMPLETED` unless that feature's `specs/<slug>/specs.md` and `requirements.md` were present and read this session.
- When `specs/<slug>/tasks.md` exists, work through its tasks top to bottom and change each task's `[ ]` to `[x]` once its Acceptance is met in code.
- When a task includes a `Refer:` line it cites stable source ids: `R#` = a point in `specs/<slug>/requirements.md`, `FR#` = a point in `specs/<slug>/specs.md` under `## Functional Requirements`, `G#` = a point in the Golden Repo Guidelines file listed in this feature's `prompt.md` (under `specs/.devx/guidance/versions/`). Before implementing such a task, OPEN the cited files, locate those ids, and implement against their full detail. The ids are an index into the authoritative detail, not a replacement for reading it. A task with no `Refer:` line is still mandatory — implement it from its Acceptance plus `specs/<slug>/specs.md` and `specs/<slug>/requirements.md`.
- Reference resolution is best-effort: if a cited id cannot be found (for example the project-level golden file is absent in a selective checkout) or the cited point clearly contradicts the task's Acceptance/intent, do NOT stop and do NOT implement the mismatched reference — fall back to the task's Acceptance plus `specs/<slug>/specs.md` and `specs/<slug>/requirements.md`, implement the task correctly, and record a one-line note in `specs/<slug>/assumptions.md` naming the unresolved/mismatched id.
- tasks.md is the ordered execution guide; `specs/<slug>/requirements.md` (the exhaustive `[R#]` checklist) and the `specs/<slug>/specs.md` Acceptance Scenarios are the AUTHORITATIVE acceptance gate. Never set a feature to `done`/`COMPLETED` until every non-blocked task in `tasks.md` is checked `[x]` AND every `requirements.md` item and every `specs.md` Acceptance Scenario is satisfied — including any not surfaced as a task.
- A `specs/.devx/change-maps/<slug>.md` file is NOT proof the feature is present; only `specs/<slug>/specs.md` is authoritative.
- When bash is available, prefer selecting the next feature via `bash specs/.devx/devx-command.sh implement-next` (or `autopilot`), which already enforces these rules.

## Golden Repo Reconciliation

- When Golden Repo artifacts exist (the Golden Repo Guidelines and Sources files listed in each feature's `prompt.md` under `specs/.devx/guidance/versions/`, or `specs/.devx/skills/ui-design/golden-ui-design-system.md`, `specs/.devx/skills/ui-design/golden-ui-design-sources.md`), you MUST read them before implementing. This is REQUIRED, not optional.
- The Golden Repo may describe a different stack than this app; do NOT apply it blindly. For each cross-cutting convention it defines (for example security/CSRF synchronizer tokens, internationalization/resource files, and the prescribed UI design system/tokens), EITHER apply it OR record it as not-applicable-to-this-stack with a one-line rationale.
- Produce a short "Golden Repo Reconciliation" note in your output: for each convention, list Applied or Not-applicable with the rationale.
- Verified local workspace patterns override Golden Repo guidance when they conflict.
- Never set a feature to `done`/`COMPLETED` when Golden Repo artifacts exist unless this read-and-reconcile was completed and the reconciliation note was produced.

## Open Questions

- Before implementing a feature, read its Open Questions (from `specs/.devx/features.json` `openQuestions[]`, or the `## Open Questions` section of `specs/<slug>/specs.md`). Open Questions are never write-only: each one must be surfaced or recorded, never silently assumed.
- Interactive mode (for example the `/implement` skill): surface the consolidated list of that feature's Open Questions and ask the user before building; apply their answers and record them.
- Unattended mode (for example `/autopilot`): do NOT stop to ask. For each Open Question, choose a reasonable assumption and RECORD it — the chosen decision plus a one-line rationale — in `specs/<slug>/assumptions.md`. Never resolve an Open Question silently.
- A feature held at status `needs-clarification` (tracker `NEEDS_CLARIFICATION`) has a BLOCKING Open Question: SKIP it like an absent folder, report it, and never modify or remove its `features.json`/`tracker.json` entry. Resolve the question and set it back to `not-started`/`PENDING` to enable it.
- Never set a feature to `done`/`COMPLETED` while a BLOCKING Open Question is unresolved and unrecorded.

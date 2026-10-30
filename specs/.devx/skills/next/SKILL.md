---
name: next
description: Show the next unimplemented feature from the spec index with its file paths and summary.
user-invocable: true
allowed-tools: Read, Bash
---

# /next — Show Next Feature to Implement

> Display the next unimplemented feature from the spec index.

## Feature Presence & Skip Rule

- A tracked feature is actionable only if its `specs/<slug>/specs.md` exists in this checkout. Features are pushed selectively, so a tracked-but-absent folder is EXPECTED and is not an error.
- If a PENDING feature's `specs/<slug>/specs.md` is missing, SKIP it: report the skipped slug, move on to the next PENDING feature, and never modify or remove its `features.json`/`tracker.json` entry.
- Never set a feature to `done`/`COMPLETED` unless that feature's `specs/<slug>/specs.md` and `requirements.md` were present and read this session.
- When `specs/<slug>/tasks.md` exists, work through its tasks top to bottom and change each task's `[ ]` to `[x]` once its Acceptance is met in code.
- When a task includes a `Refer:` line it cites stable source ids: `R#` = a point in `specs/<slug>/requirements.md`, `FR#` = a point in `specs/<slug>/specs.md` under `## Functional Requirements`, `G#` = a point in the Golden Repo Guidelines file listed in this feature's `prompt.md` (under `specs/.devx/guidance/versions/`). Before implementing such a task, OPEN the cited files, locate those ids, and implement against their full detail. The ids are an index into the authoritative detail, not a replacement for reading it. A task with no `Refer:` line is still mandatory — implement it from its Acceptance plus `specs/<slug>/specs.md` and `specs/<slug>/requirements.md`.
- Reference resolution is best-effort: if a cited id cannot be found (for example the project-level golden file is absent in a selective checkout) or the cited point clearly contradicts the task's Acceptance/intent, do NOT stop and do NOT implement the mismatched reference — fall back to the task's Acceptance plus `specs/<slug>/specs.md` and `specs/<slug>/requirements.md`, implement the task correctly, and record a one-line note in `specs/<slug>/assumptions.md` naming the unresolved/mismatched id.
- tasks.md is the ordered execution guide; `specs/<slug>/requirements.md` (the exhaustive `[R#]` checklist) and the `specs/<slug>/specs.md` Acceptance Scenarios are the AUTHORITATIVE acceptance gate. Never set a feature to `done`/`COMPLETED` until every non-blocked task in `tasks.md` is checked `[x]` AND every `requirements.md` item and every `specs.md` Acceptance Scenario is satisfied — including any not surfaced as a task.
- A `specs/.devx/change-maps/<slug>.md` file is NOT proof the feature is present; only `specs/<slug>/specs.md` is authoritative.
- When bash is available, prefer selecting the next feature via `bash specs/.devx/devx-command.sh implement-next` (or `autopilot`), which already enforces these rules.

## Open Questions (surface and resolve)

- Before implementing a feature, read its Open Questions (from `specs/.devx/features.json` `openQuestions[]`, or the `## Open Questions` section of `specs/<slug>/specs.md`). Open Questions are never write-only: each one must be surfaced or recorded, never silently assumed.
- Interactive mode (for example the `/implement` skill): surface the consolidated list of that feature's Open Questions and ask the user before building; apply their answers and record them.
- Unattended mode (for example `/autopilot`): do NOT stop to ask. For each Open Question, choose a reasonable assumption and RECORD it — the chosen decision plus a one-line rationale — in `specs/<slug>/assumptions.md`. Never resolve an Open Question silently.
- A feature held at status `needs-clarification` (tracker `NEEDS_CLARIFICATION`) has a BLOCKING Open Question: SKIP it like an absent folder, report it, and never modify or remove its `features.json`/`tracker.json` entry. Resolve the question and set it back to `not-started`/`PENDING` to enable it.
- Never set a feature to `done`/`COMPLETED` while a BLOCKING Open Question is unresolved and unrecorded.

## Workflow

1. Read the feature index:
   - Open `specs/.devx/features.json`
   - Open `specs/.devx/tracker.json` when present
   - Parse the features array

2. Find the next feature:
   - Prefer the first tracker entry with `"status": "PENDING"`
   - Fall back to the first feature with `"status": "not-started"`
   - SKIP any candidate whose `specs/<slug>/specs.md` is not present in this checkout (report the skipped slug and leave its tracking entry untouched); continue to the next present candidate
   - SKIP any candidate whose status is `needs-clarification` (tracker `NEEDS_CLARIFICATION`) — it has an unresolved BLOCKING Open Question; report it and continue to the next candidate
   - If all present features are done, report that

3. Display the feature summary:
   - Title and slug
   - Number of user stories and total story points
   - File paths (specs.md, requirements.md, tdd-tests.md if applicable, prompt.md)
   - Brief description from the first few lines of specs.md

4. Ask the user if they want to start implementing it:
   - If yes, proceed with the /implement workflow
   - If no, show the full feature list with statuses

## Output Format

```
## Next Feature: <Title>

- **Slug:** <slug>
- **Stories:** <count> (<total points> story points)
- **Status:** not-started

### Files
- specs/<slug>/specs.md
- specs/<slug>/requirements.md
- specs/<slug>/prompt.md

### Summary
<First paragraph from specs.md>

Ready to implement? Use /implement to start.
```

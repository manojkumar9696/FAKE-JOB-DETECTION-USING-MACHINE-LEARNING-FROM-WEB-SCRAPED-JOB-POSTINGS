---
name: implement
description: Implement a feature following the Spec-Driven Development workflow. Reads specs, implementation acceptance requirements, and TDD tests to guide implementation.
user-invocable: true
argument-hint: "[feature-slug]"
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# /implement — Implement a Feature

> Pick a feature from the spec index, read its full specification, and generate an implementation plan.

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

## Golden Repo Guidance (read and reconcile)

- When Golden Repo artifacts exist (the Golden Repo Guidelines and Sources files listed in each feature's `prompt.md` under `specs/.devx/guidance/versions/`, or `specs/.devx/skills/ui-design/golden-ui-design-system.md`, `specs/.devx/skills/ui-design/golden-ui-design-sources.md`), you MUST read them before implementing. This is REQUIRED, not optional.
- The Golden Repo may describe a different stack than this app; do NOT apply it blindly. For each cross-cutting convention it defines (for example security/CSRF synchronizer tokens, internationalization/resource files, and the prescribed UI design system/tokens), EITHER apply it OR record it as not-applicable-to-this-stack with a one-line rationale.
- Produce a short "Golden Repo Reconciliation" note in your output: for each convention, list Applied or Not-applicable with the rationale.
- Verified local workspace patterns override Golden Repo guidance when they conflict.
- Never set a feature to `done`/`COMPLETED` when Golden Repo artifacts exist unless this read-and-reconcile was completed and the reconciliation note was produced.

## Open Questions (surface and resolve)

- Before implementing a feature, read its Open Questions (from `specs/.devx/features.json` `openQuestions[]`, or the `## Open Questions` section of `specs/<slug>/specs.md`). Open Questions are never write-only: each one must be surfaced or recorded, never silently assumed.
- Interactive mode (for example the `/implement` skill): surface the consolidated list of that feature's Open Questions and ask the user before building; apply their answers and record them.
- Unattended mode (for example `/autopilot`): do NOT stop to ask. For each Open Question, choose a reasonable assumption and RECORD it — the chosen decision plus a one-line rationale — in `specs/<slug>/assumptions.md`. Never resolve an Open Question silently.
- A feature held at status `needs-clarification` (tracker `NEEDS_CLARIFICATION`) has a BLOCKING Open Question: SKIP it like an absent folder, report it, and never modify or remove its `features.json`/`tracker.json` entry. Resolve the question and set it back to `not-started`/`PENDING` to enable it.
- Never set a feature to `done`/`COMPLETED` while a BLOCKING Open Question is unresolved and unrecorded.

## Workflow

0. Run tracking preflight:
   - Execute `bash specs/.devx/validate-tracking.sh`
   - If it reports uncommitted or unpushed changes, stop and ask the user to commit/push or stash first

1. Read the feature index:
   - Open `specs/.devx/features.json`
   - Open `specs/.devx/tracker.json`
   - Find the first feature whose tracker entry has `"status": "PENDING"`, or ask the user which feature to implement
   - SKIP any PENDING feature whose `specs/<slug>/specs.md` is not present in this checkout (report the skipped slug and leave its tracking entry untouched); pick the next present PENDING feature instead
   - SKIP any feature whose status is `needs-clarification` (tracker `NEEDS_CLARIFICATION`) — it has an unresolved BLOCKING Open Question; report it and pick the next present PENDING feature
   - Note the feature slug

2. Read portable handoff context:
   - Open `specs/.devx/project-context.md`
   - Open `specs/.devx/current-state.md`
   - Open `specs/.devx/implementation-check.md`
   - Open `specs/.devx/change-maps/<slug>.md`
   - When present, you MUST read the Golden Repo Guidelines and Sources files listed in this feature's `prompt.md` (under `specs/.devx/guidance/versions/`) and reconcile each convention (apply, or mark not-applicable-to-this-stack with a rationale) — see the Golden Repo Guidance rule above
   - If implementation-check is FAIL, stop and ask for discovery/context repair before coding

3. Inspect existing code before planning:
   - Use the change map and current-state summary to find existing files, routes, components, services, tests, and shared modules
   - Verify suggested paths in the actual workspace before editing
   - Existing code patterns override generic guidance when they conflict

4. Read the full specification:
   - Open `specs/<slug>/specs.md`
   - Understand the Summary, Key Features, Functional Requirements, and User Scenarios
   - Read this feature's Open Questions (`specs/.devx/features.json` `openQuestions[]`, or the `## Open Questions` section). Surface the consolidated list to the user and ask before building; apply and record their answers. In an unattended run, record the chosen assumption + rationale per question in `specs/<slug>/assumptions.md` instead — never assume silently
   - Do not skip any section

5. Read the implementation acceptance checklist:
   - Open `specs/<slug>/requirements.md`
   - This is the implementation acceptance checklist — every item must be satisfied
   - If this file contains spec-quality PASS/FAIL validation results instead of implementation acceptance criteria, stop and ask the user to regenerate or repair the spec bundle


6. Create an implementation plan:
   - Break the work into concrete steps based on the implementation acceptance checklist
   - Identify existing files to modify before proposing new files
   - Identify dependencies between checklist items
   - Present the plan to the user for approval

7. Implement one checklist item at a time:
   - Follow the spec exactly — do not add features not described
   - After each checklist item, verify it against requirements.md

8. After all checklist items are complete:
   - Review the full implementation acceptance checklist in requirements.md
   - Verify all user scenarios from specs.md work correctly
   - Mark-done preconditions: only proceed if this feature's `specs/<slug>/specs.md` and `requirements.md` were present and read this session; when Golden Repo artifacts exist, the Golden Repo Reconciliation note was produced; and every BLOCKING Open Question is resolved and recorded in `specs/<slug>/assumptions.md`
   - Update features.json status to "done" and tracker.json status to "COMPLETED"

## Rules

- **Do not deviate from the specification.** If the spec is wrong, update it first.
- **One checklist item at a time.** Complete and validate before moving on.
- **The implementation acceptance checklist is the source of truth** for what "done" means.
- **Inspect existing code first.** Do not create parallel implementations when an existing module should be extended.
- **Use local guidance only.** Do not rely on Astra or live Golden Repo access during IDE implementation.
- **No gold-plating.** Do not add features beyond what the spec describes.

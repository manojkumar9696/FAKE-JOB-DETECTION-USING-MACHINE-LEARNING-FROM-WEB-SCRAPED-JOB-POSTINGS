---
name: validate
description: Validate current code against the implementation acceptance checklist for a feature. Produces a pass/fail report.
user-invocable: true
argument-hint: "[feature-slug]"
allowed-tools: Read, Grep, Glob, Bash
---

# /validate — Validate Implementation Against Acceptance Checklist

> Check the current code against the implementation acceptance checklist for a specific feature.

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
   - If it reports uncommitted changes, unpushed completed work, or tracking inconsistencies, stop before validation

1. Identify the feature to validate:
   - Ask the user which feature to validate, or use the most recently implemented one
   - Find it in `specs/.devx/features.json` by slug, title, or ID
   - If the feature's `specs/<slug>/specs.md` is not present in this checkout, do NOT validate or mark it done — report it as skipped (absent folder) and leave its tracking entry untouched
   - If the feature's status is `needs-clarification` (tracker `NEEDS_CLARIFICATION`), it has an unresolved BLOCKING Open Question: do NOT validate or mark it done — report it as skipped (needs clarification) and leave its tracking entry untouched
   - A feature marked `done`/`COMPLETED` must FAIL validation if any BLOCKING Open Question in `specs/.devx/features.json` (`openQuestions[]`) is unresolved and has no recorded decision in `specs/<slug>/assumptions.md`
   - When Golden Repo artifacts exist, confirm a "Golden Repo Reconciliation" note was produced for this feature; if it is missing, mark validation FAIL until the guidance is read and reconciled

2. Read the implementation acceptance checklist:
   - Open `specs/<slug>/requirements.md`
   - This contains every implementation acceptance item that must be satisfied
   - If this file contains spec-quality PASS/FAIL validation results instead of implementation acceptance criteria, stop and report that the spec bundle must be regenerated or repaired

3. Read portable handoff context:
   - Open `specs/.devx/project-context.md`
   - Open `specs/.devx/current-state.md`
   - Open `specs/.devx/implementation-check.md`
   - Open `specs/.devx/change-maps/<slug>.md`
   - If present, open the Golden Repo Guidelines file listed in this feature's `prompt.md` (under `specs/.devx/guidance/versions/`)

4. Read the specification for context:
   - Open `specs/<slug>/specs.md`
   - Understand the expected behavior and edge cases

5. For each item in the implementation acceptance checklist:
   - Search the codebase to verify it is implemented
   - Check that the implementation matches the spec (not just that code exists)
   - Check that implementation follows project-context and current-state constraints
   - Look for edge cases described in the spec
   - Mark each checklist item as PASS or FAIL with a brief explanation

6. Generate a validation report:
   - List each checklist item with its PASS/FAIL status
   - For any FAIL items, explain what is missing or incorrect
   - Provide a summary: X of Y implementation acceptance items satisfied
   - If all pass, suggest updating features.json status to "done" and tracker.json status to "COMPLETED"

## Output Format

```
## Validation Report — <Feature Title>

| # | Acceptance Item | Status | Notes |
|---|-------------|--------|-------|
| 1 | <acceptance checklist item> | PASS/FAIL | <explanation> |
| ... | ... | ... | ... |

**Result: X/Y implementation acceptance items satisfied**
```

## Rules

- Be thorough — check actual implementation, not just file existence
- Look at edge cases and error handling described in the spec
- Do not mark a checklist item as PASS if it is only partially implemented
- Validate against local project context and Golden Repo guidance when present
- If the feature has TDD tests, verify they all pass as well

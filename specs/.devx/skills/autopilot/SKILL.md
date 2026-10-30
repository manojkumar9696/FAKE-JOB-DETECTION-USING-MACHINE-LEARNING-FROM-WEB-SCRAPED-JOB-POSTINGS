---
name: autopilot
description: Automatically implement all remaining features end-to-end. Picks the next feature, implements it, validates, marks done, and repeats until all features are complete.
user-invocable: true
argument-hint: "[max-features]"
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# /autopilot — Implement All Features Automatically

> Chains the full SDD workflow in a loop: pick next → implement → validate → mark done → repeat.

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

## Arguments

- `$ARGUMENTS` — optional max number of features to implement in this session (default: all remaining)

## Workflow

For each unimplemented feature, execute these phases in sequence:

### Phase 0 — Tracking Preflight
1. Run `bash specs/.devx/validate-tracking.sh`
2. If it reports blocking app/spec changes or unpushed completed work, stop and ask the user to commit/push or stash first
3. If only setup files or generated DevX bookkeeping files changed, continue; those files can be committed with the completed feature

### Phase 1 — Pick Next Feature
1. Read `specs/.devx/features.json` and `specs/.devx/tracker.json`
2. Find the first feature whose tracker entry has `"status": "PENDING"`. SKIP any PENDING feature whose `specs/<slug>/specs.md` is not present in this checkout — features are pushed selectively, so a not-present folder is expected and is not an error. Note the skipped slug in your output, never modify or remove its `features.json`/`tracker.json` entry, and move on to the next present PENDING feature. Also SKIP any feature whose status is `needs-clarification` (tracker `NEEDS_CLARIFICATION`) — it has an unresolved BLOCKING Open Question; report it and leave its entry untouched
3. If no present features remain, report completion and stop
4. Announce: "Starting feature: <title> (<N> of <total> remaining)"

### Phase 2 — Read Specs
1. Read `specs/.devx/project-context.md`
2. Read `specs/.devx/current-state.md`
3. Read `specs/.devx/implementation-check.md`; stop if status is FAIL
4. Read `specs/.devx/change-maps/<slug>.md`
5. When present, you MUST read the Golden Repo Guidelines and Sources files listed in this feature's `prompt.md` (under `specs/.devx/guidance/versions/`) (and the ui-design golden files when present) and reconcile each convention — apply it, or mark it not-applicable-to-this-stack with a rationale — per the Golden Repo Guidance rule above
6. Read `specs/<slug>/specs.md` — understand what to build
6a. Read this feature's Open Questions (`specs/.devx/features.json` `openQuestions[]`, or the `## Open Questions` section of `specs/<slug>/specs.md`). This is an unattended run: for each Open Question choose a reasonable assumption and RECORD the decision plus a one-line rationale in `specs/<slug>/assumptions.md` before implementing. Never resolve an Open Question silently. (Blocking questions are already held at `needs-clarification` and skipped in Phase 1.)
7. Read `specs/<slug>/requirements.md` — the implementation acceptance checklist
   - If it contains spec-quality PASS/FAIL validation results instead of implementation acceptance criteria, stop and ask the user to regenerate or repair the spec bundle

### Phase 2.5 — Inspect Existing Code
1. Use the change map and current-state summary to locate existing files, routes, components, services, tests, and shared modules
2. Verify suggested paths in the actual workspace before editing
3. Prefer extending existing code over creating parallel implementations
4. Existing code patterns override generic guidance when they conflict

### Phase 3 — Implement
Implement each implementation acceptance item from requirements.md one at a time:

1. Read the implementation acceptance item
2. Write the code to satisfy it
3. Verify it works
4. Move to the next checklist item

### Phase 4 — Validate
1. Go through every item in `specs/<slug>/requirements.md`
2. Verify each implementation acceptance item is satisfied in the code
3. Check edge cases from specs.md

If validation fails:
- Fix the failing criteria
- Re-validate
- Do not proceed until all criteria pass

### Phase 5 — Mark Done
0. Mark-done preconditions — do NOT mark this feature done unless ALL hold: its `specs/<slug>/specs.md` and `requirements.md` were present and read this session; validation passed; when Golden Repo artifacts exist, the "Golden Repo Reconciliation" note was produced; and every BLOCKING Open Question is resolved and recorded in `specs/<slug>/assumptions.md` (a feature at `needs-clarification` must never be marked done). If a precondition fails, stop and report instead of updating status
1. Read `specs/.devx/features.json` and `specs/.devx/tracker.json`
2. Update this feature's `"status"` from `"not-started"` to `"done"` in features.json
3. Update this feature's tracker status from `"PENDING"` to `"COMPLETED"`, refresh `updatedAt`, and increment `trackerVersion`
4. Write both files back
5. Announce: "Completed: <title> ✓"

### Phase 6 — Continue or Stop
1. Check if max features limit reached (from `$ARGUMENTS`)
2. If more features remain and limit not reached → go to Phase 1
3. If all features done → announce completion summary
4. If limit reached → announce progress and remaining count

## Output Between Features

After each feature, show a brief status:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Completed: <Feature Title>
  Requirements: 8/8 passed
  Next: <Next Feature Title> (N remaining)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Final Summary

When all features are done (or limit reached), show:
```
══════════════════════════════════════
  Autopilot Complete
  Features implemented: X / Y
  Total implementation acceptance items satisfied: N
  Remaining: Z features
══════════════════════════════════════
```

## Rules

- **Never skip validation.** Every feature must pass all implementation acceptance items before moving on.
- **Stop on repeated failure.** If a feature fails validation 3 times, stop and report the issue.
- **Do not modify specs.** If a spec seems wrong, stop and ask the user.
- **Do not rely on Astra or live Golden Repo access.** Use only the local files under `specs/.devx/`.
- **Inspect existing code before editing.** Change maps are advisory until verified in the IDE workspace.
- **Commit after each feature.** Keep changes atomic and reviewable.
- **No gold-plating.** Implement exactly what the spec describes, nothing more.

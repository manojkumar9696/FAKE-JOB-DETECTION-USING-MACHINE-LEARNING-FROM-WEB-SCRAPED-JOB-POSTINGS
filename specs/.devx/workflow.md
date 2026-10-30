# Development Workflow

> Spec-Driven Development (SDD) workflow guide.

## How to Implement a Feature

### Step 1 — Understand the Spec
1. Open the feature's `specs.md`
2. Read the Summary, Key Features, and Functional Requirements
3. Review the User Scenarios for expected behavior

### Step 2 — Review Implementation Acceptance Checklist
1. Open `requirements.md`
2. This is your implementation acceptance checklist
3. Every item must pass before the feature is considered complete
4. If this file contains spec-quality PASS/FAIL validation results instead of implementation acceptance criteria, regenerate or repair the spec bundle before coding

### Step 3 — Implement
1. Open `prompt.md` — paste it into your AI assistant for guided implementation
2. For microservices, review `specs/.devx/feature-routing.json` and `specs/.devx/workspace-repos.json`
3. Follow the specification exactly — do not add features not in the spec
4. Implement one checklist item at a time

### Step 4 — Validate
1. Go through `requirements.md` line by line
2. Check each implementation acceptance item
3. Ensure all edge cases from the spec are handled

### Step 5 — Submit
1. Create a PR with the implementation
2. Reference the spec file in the PR description
3. Include the implementation acceptance checklist with pass/fail status

## Rules

- **Do not deviate from the spec.** If the spec is wrong, update the spec first.
- **One feature at a time.** Complete and validate before moving to the next.
- **The implementation acceptance checklist is the source of truth** for what "done" means.
- **Every PR must reference** the spec and requirements it implements.

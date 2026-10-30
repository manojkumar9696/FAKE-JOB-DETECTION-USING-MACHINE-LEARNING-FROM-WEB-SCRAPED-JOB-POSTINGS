#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURES_FILE="$SCRIPT_DIR/features.json"
TRACKER_FILE="$SCRIPT_DIR/tracker.json"
TRACKING_VALIDATOR="$SCRIPT_DIR/validate-tracking.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash specs/.devx/devx-command.sh autopilot [max-features]
  bash specs/.devx/devx-command.sh implement-next
  bash specs/.devx/devx-command.sh implement-feature <feature-slug>
  bash specs/.devx/devx-command.sh validate-feature <feature-slug>
USAGE
}

ensure_features_file() {
  if [ ! -f "$FEATURES_FILE" ]; then
    echo "features.json not found at $FEATURES_FILE" >&2
    exit 1
  fi
}

run_tracking_preflight() {
  if [ -f "$TRACKING_VALIDATOR" ]; then
    bash "$TRACKING_VALIDATOR"
  else
    echo "Tracking validator not found at $TRACKING_VALIDATOR" >&2
    exit 1
  fi
}

feature_field() {
  local slug="$1"
  local field="$2"
  node - "$FEATURES_FILE" "$slug" "$field" <<'NODE'
const fs = require("fs");
const [file, slug, field] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(file, "utf8"));
const feature = (data.features || []).find((item) => item.slug === slug);
if (!feature) process.exit(2);
const value = field.split(".").reduce((acc, key) => acc && acc[key], feature);
if (value !== undefined && value !== null) process.stdout.write(String(value));
NODE
}

resolve_next_slug() {
  ensure_features_file
  node - "$FEATURES_FILE" "$TRACKER_FILE" <<'NODE'
const fs = require("fs");
const path = require("path");
const [featuresFile, trackerFile] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(featuresFile, "utf8"));
const tracker = fs.existsSync(trackerFile)
  ? JSON.parse(fs.readFileSync(trackerFile, "utf8"))
  : { tasks: {} };
// featuresFile lives at <repo>/specs/.devx/features.json, so the repo root is
// two levels up from its directory. Feature file paths are repo-root relative.
const repoRoot = path.resolve(path.dirname(featuresFile), "..", "..");
const isPending = (item) => {
  const taskId = item.taskId || `FEATURE-${item.id}`;
  const taskStatus = tracker.tasks?.[taskId]?.status;
  return taskStatus ? taskStatus === "PENDING" : item.status === "not-started";
};
// Only offer features whose folder was actually pushed to this checkout. A
// feature tracked but not present is skipped (not an error) so implement-next
// moves on to the next present PENDING feature.
const isPresent = (item) => {
  const specs = (item.files && item.files.specs) || `specs/${item.slug}/specs.md`;
  return fs.existsSync(path.join(repoRoot, specs));
};
const pending = (data.features || []).filter(isPending);
const feature = pending.find(isPresent);
if (!feature) {
  const skipped = pending.filter((item) => !isPresent(item)).map((item) => item.slug);
  if (skipped.length > 0) {
    console.error("Skipped PENDING feature(s) not present in this checkout: " + skipped.join(", "));
  }
  console.error("No PENDING feature with a present specs/<slug>/ folder was found.");
  process.exit(1);
}
process.stdout.write(feature.slug);
NODE
}

assert_feature_exists() {
  local slug="$1"
  if ! feature_field "$slug" "title" >/dev/null 2>&1; then
    echo "Feature not found in features.json: $slug" >&2
    exit 1
  fi
}

emit_implementation_prompt() {
  local slug="$1"
  assert_feature_exists "$slug"
  local title specs_path requirements_path prompt_path tdd_path repo_root feature_dir
  title="$(feature_field "$slug" "title")"
  specs_path="$(feature_field "$slug" "files.specs")"
  requirements_path="$(feature_field "$slug" "files.requirements")"
  prompt_path="$(feature_field "$slug" "files.prompt")"
  tdd_path="$(feature_field "$slug" "files.tddTests" || true)"

  # A feature can be tracked without its specs/<slug>/ folder being pushed to
  # this checkout (features are pushed selectively). Skip gracefully with a
  # distinct non-zero exit so the caller/wrapper aborts instead of emitting a
  # prompt that points at missing files. The tracking entry is left untouched.
  if [ -z "$specs_path" ]; then
    specs_path="specs/$slug/specs.md"
  fi
  repo_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
  feature_dir="$(dirname "$repo_root/$specs_path")"
  if [ ! -d "$feature_dir" ]; then
    printf '%s\n' "Feature \"$slug\" folder is not present in this checkout (specs/$slug/ missing); skipping." >&2
    printf '%s\n' "Its features.json/tracker.json entry is unchanged and will resume once the folder is pushed." >&2
    exit 10
  fi

  cat <<PROMPT
Read AGENTS.md and the local handoff files first:
- specs/.devx/project-context.md
- specs/.devx/current-state.md
- specs/.devx/implementation-check.md
- specs/.devx/change-maps/$slug.md
- the Golden Repo Guidelines file listed in this feature's prompt.md (under specs/.devx/guidance/versions/) when present

Selected feature slug: $slug
Selected feature title: $title

You must only implement this selected feature:
- Specs: $specs_path
- Requirements: $requirements_path
- Prompt: $prompt_path


Strict guardrails:
- Do not infer the feature from the currently open editor tab.
- Do not implement DevX tooling, init.sh, specs generation, workspace bootstrap, or unrelated specs unless this selected feature explicitly requires it.
- Do not modify other feature folders except when a shared implementation file must change to satisfy this feature.
- Inspect existing code paths from the change map before creating an implementation plan.
- When Golden Repo guidance exists (specs/.devx/guidance/ or the specs/.devx/skills/ui-design/ golden files), you MUST read it and reconcile each cross-cutting convention (e.g. CSRF synchronizer tokens, i18n/resource files, the UI design system/tokens): apply it, or mark it not-applicable-to-this-stack with a one-line rationale, and include a short "Golden Repo Reconciliation" note. Verified local patterns override golden guidance on conflict.
- Read this feature's Open Questions (specs/.devx/features.json openQuestions[], or the "## Open Questions" section in $specs_path). Surface the consolidated list and ask the human user for a decision before coding, apply their answer, and record the decision plus a one-line rationale in specs/$slug/assumptions.md — never assume silently or invent behavior. A feature held at status needs-clarification (tracker NEEDS_CLARIFICATION) has an unresolved blocking question and must be resolved and re-enabled (set back to not-started/PENDING) before implementing.
- Create an implementation plan before editing.
- Implement one item from the implementation acceptance checklist at a time.
- If $requirements_path contains spec-quality PASS/FAIL validation results instead of implementation acceptance criteria, stop and ask the user to regenerate or repair the spec bundle.
- Validate the implementation against every item in $requirements_path.

- If all implementation acceptance items pass (and, when Golden Repo artifacts exist, the Golden Repo Reconciliation note was produced, and every blocking Open Question is resolved and recorded in specs/$slug/assumptions.md), update only this feature's status in specs/.devx/features.json to "done" and its entry in specs/.devx/tracker.json to "COMPLETED" with a fresh updatedAt timestamp.

Now read the selected feature files and begin with the implementation plan.
PROMPT
}

emit_validation_prompt() {
  local slug="$1"
  assert_feature_exists "$slug"
  local title specs_path requirements_path repo_root feature_dir
  title="$(feature_field "$slug" "title")"
  specs_path="$(feature_field "$slug" "files.specs")"
  requirements_path="$(feature_field "$slug" "files.requirements")"

  # Same selective-push guard as emit_implementation_prompt: if the feature's
  # folder was not pushed to this checkout, skip with a distinct non-zero exit so
  # the wrapper aborts instead of validating against missing files. Tracking
  # entry is left untouched.
  if [ -z "$specs_path" ]; then
    specs_path="specs/$slug/specs.md"
  fi
  repo_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
  feature_dir="$(dirname "$repo_root/$specs_path")"
  if [ ! -d "$feature_dir" ]; then
    printf '%s\n' "Feature \"$slug\" folder is not present in this checkout (specs/$slug/ missing); skipping validation." >&2
    printf '%s\n' "Its features.json/tracker.json entry is unchanged and will resume once the folder is pushed." >&2
    exit 10
  fi

  cat <<PROMPT
Read AGENTS.md and the local handoff files first:
- specs/.devx/project-context.md
- specs/.devx/current-state.md
- specs/.devx/implementation-check.md
- specs/.devx/change-maps/$slug.md
- the Golden Repo Guidelines file listed in this feature's prompt.md (under specs/.devx/guidance/versions/) when present

Validate this selected feature only:
- Slug: $slug
- Title: $title
- Specs: $specs_path
- Requirements: $requirements_path

Validation rules:
- Check every implementation acceptance item in $requirements_path.
- If $requirements_path contains spec-quality PASS/FAIL validation results instead of implementation acceptance criteria, stop and report that the spec bundle must be regenerated or repaired.
- Verify behavior in actual implementation code, not just file existence.
- Verify the implementation follows local project context and existing-code constraints.
- Report PASS/FAIL for each implementation acceptance item.
- Do not mark the feature done unless every implementation acceptance item passes.
- Do not validate unrelated features.
PROMPT
}

emit_autopilot_prompt() {
  local max_features="${1:-all}"
  cat <<PROMPT
Read AGENTS.md and local handoff files before each feature:
- specs/.devx/project-context.md
- specs/.devx/current-state.md
- specs/.devx/implementation-check.md
- specs/.devx/change-maps/<slug>.md
- the Golden Repo Guidelines file listed in this feature's prompt.md (under specs/.devx/guidance/versions/) when present

Run DevX autopilot code generation.

Scope:
- Maximum features this session: $max_features
- Source of truth: specs/.devx/features.json, specs/.devx/tracker.json, and each specs/<slug>/ folder

Mandatory loop:
1. Pick the first feature whose tracker status is "PENDING". Skip any feature whose specs/<slug>/ folder is not present in this checkout (features are pushed selectively) - note the skipped slug in your output and move on to the next PENDING feature. A not-present folder is expected and is not an error; never modify or remove its features.json/tracker.json entry. Also skip any feature whose status is needs-clarification (tracker NEEDS_CLARIFICATION) - it has an unresolved blocking Open Question; report it and leave its entry untouched.
2. Read its local handoff context, change map, specs.md, requirements.md, prompt.md, and tdd-tests.md when present.
2a. When Golden Repo guidance exists (specs/.devx/guidance/ or the specs/.devx/skills/ui-design/ golden files), you MUST read it and reconcile each cross-cutting convention (e.g. CSRF synchronizer tokens, i18n/resource files, the UI design system/tokens): apply it, or mark it not-applicable-to-this-stack with a one-line rationale, and include a short "Golden Repo Reconciliation" note. Verified local patterns override golden guidance on conflict.
3. Read this feature's Open Questions (specs/.devx/features.json openQuestions[], or the "## Open Questions" section in that feature's specs.md). This is an unattended run: do NOT pause to ask. For each Open Question, choose a reasonable assumption and record the decision plus a one-line rationale in specs/<slug>/assumptions.md before coding - never assume silently or invent behavior. (Blocking questions are already held at needs-clarification and skipped in step 1.)
4. Inspect existing code before editing and verify change-map suggestions in the actual workspace.
5. Implement only that feature, one implementation acceptance item at a time.
6. Validate every implementation acceptance item against actual implementation code.
7. If requirements.md contains spec-quality PASS/FAIL validation results instead of implementation acceptance criteria, stop and ask the user to regenerate or repair the spec bundle.
8. If all implementation acceptance items pass (and, when Golden Repo artifacts exist, the Golden Repo Reconciliation note was produced, and every blocking Open Question is resolved and recorded in specs/<slug>/assumptions.md), mark only that feature as "done" in specs/.devx/features.json and "COMPLETED" in specs/.devx/tracker.json.
9. Stop if tracking becomes dirty, validation fails repeatedly, or the max feature limit is reached.

Do not start new work while uncommitted or unpushed code or tracking changes exist.
PROMPT
}

main() {
  local command="${1:-}"
  local slug="${2:-}"
  case "$command" in
    autopilot|implement-next|implement-feature|validate-feature) run_tracking_preflight ;;
    *) ;;
  esac
  case "$command" in
    autopilot)
      emit_autopilot_prompt "${slug:-all}"
      ;;
    implement-next)
      slug="$(resolve_next_slug)"
      emit_implementation_prompt "$slug"
      ;;
    implement-feature)
      if [ -z "$slug" ]; then
        echo "Missing feature slug." >&2
        usage
        exit 1
      fi
      emit_implementation_prompt "$slug"
      ;;
    validate-feature)
      if [ -z "$slug" ]; then
        echo "Missing feature slug." >&2
        usage
        exit 1
      fi
      emit_validation_prompt "$slug"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"

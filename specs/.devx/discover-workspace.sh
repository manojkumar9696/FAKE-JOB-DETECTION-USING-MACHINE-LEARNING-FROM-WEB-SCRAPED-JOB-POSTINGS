#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
DEVX_DIR="$ROOT_DIR/specs/.devx"
WORKSPACE_REPOS_FILE="$DEVX_DIR/workspace-repos.json"
WORKSPACE_CONTEXT_FILE="$DEVX_DIR/workspace-context.md"
REPO_PLANS_FILE="$DEVX_DIR/repo-plans.json"
PROJECT_CONTEXT_FILE="$DEVX_DIR/project-context.md"
CURRENT_STATE_FILE="$DEVX_DIR/current-state.md"
IMPLEMENTATION_CHECK_FILE="$DEVX_DIR/implementation-check.md"
CHANGE_MAPS_DIR="$DEVX_DIR/change-maps"
MAX_DEPTH=4

mkdir -p "$DEVX_DIR"
mkdir -p "$CHANGE_MAPS_DIR"

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

json_escape() {
  if command -v node >/dev/null 2>&1; then
    node -e 'process.stdout.write(JSON.stringify(process.argv[1] || "").slice(1, -1))' "$1"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json,sys; s=sys.argv[1] if len(sys.argv)>1 else ""; print(json.dumps(s)[1:-1], end="")' "$1"
  else
    printf '%s' "$1" | sed 's/\/\\/g; s/"/\"/g; s/	/\t/g; s/
/\r/g; s/
/\n/g'
  fi
}

csv_escape() {
  printf '%s' "$1" | sed 's/|/\\|/g'
}

join_csv() {
  local IFS=", "
  printf '%s' "$*"
}

is_discovery_excluded_path() {
  local candidate="$1"
  local normalized
  normalized="$(printf '%s' "$candidate" | sed 's#^./##; s#//*#/#g')"

  case "$normalized" in
    specs/.devx|specs/.devx/*|.claude|.claude/*|.cursor|.cursor/*|.kiro|.kiro/*|.vscode|.vscode/*|.github|.github/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

collect_repo_roots() {
  if find "$ROOT_DIR" -maxdepth 1 -name ".git" -print -quit | grep -q .; then
    printf '.
'
  fi

  find "$ROOT_DIR" -mindepth 1 -maxdepth "$MAX_DEPTH" -type d \
    \( -name .git -o -name node_modules -o -name dist -o -name build -o -name coverage -o -name .next -o -name .turbo -o -name .pnpm-store -o -name .yarn -o -name .idea -o -name .vscode -o -name __pycache__ \) -prune -o \
    \( -name .git -o -name package.json -o -name pnpm-workspace.yaml -o -name pom.xml -o -name build.gradle -o -name build.gradle.kts -o -name Cargo.toml -o -name go.mod -o -name requirements.txt -o -name Dockerfile -o -name '*.csproj' -o -name '*.sln' \) -print |
    while IFS= read -r marker; do
      repo_dir="$(dirname "$marker")"
      repo_rel="${repo_dir#"$ROOT_DIR"/}"
      if [ "$repo_dir" = "$ROOT_DIR" ]; then
        repo_rel="."
      fi
      if ! is_discovery_excluded_path "$repo_rel"; then
        printf '%s
' "$repo_dir"
      fi
    done | sort -u
}

list_repo_files() {
  local repo_abs="$1"
  local repo_rel
  repo_rel="${repo_abs#"$ROOT_DIR"/}"
  if [ "$repo_abs" = "$ROOT_DIR" ]; then
    repo_rel="."
  fi
  if is_discovery_excluded_path "$repo_rel"; then
    return 0
  fi
  find "$repo_abs" -mindepth 1 -maxdepth 4 \
    \( -type d \( -name .git -o -name node_modules -o -name dist -o -name build -o -name coverage -o -name .next -o -name .turbo -o -name .pnpm-store -o -name .yarn -o -name .idea -o -name .vscode -o -name __pycache__ \) -prune \) -o \
    -type f -print |
    sed "s#^$repo_abs/##" |
    sort -u
}

detect_repo_type() {
  local rel_path="$1"
  local files="$2"
  local lower_rel
  lower_rel="$(printf '%s' "$rel_path" | tr '[:upper:]' '[:lower:]')"

  if printf '%s
' "$lower_rel" | grep -Eq '(^|/)(lib|libs|shared|packages/shared|packages/ui)(/|$)'; then
    printf 'shared-lib'
  elif printf '%s
' "$files" | grep -Eiq '(src/pages/|app/.+/page\.|routes\.tsx|router|page\.(tsx|jsx|ts|js)$)'; then
    printf 'ui'
  elif printf '%s
' "$files" | grep -Eiq '(worker|job|queue|consumer|processor)'; then
    printf 'worker'
  elif printf '%s
' "$files" | grep -Eiq '(controller|route|routes|/api/|api/)'; then
    printf 'api'
  elif printf '%s
' "$files" | grep -Eiq '\.(tf|bicep)$'; then
    printf 'infra'
  else
    printf 'unknown'
  fi
}

detect_languages() {
  local files="$1"
  local langs=()

  printf '%s
' "$files" | grep -Eq '\.(ts|tsx)$' && langs+=("TypeScript")
  printf '%s
' "$files" | grep -Eq '\.(js|jsx)$' && langs+=("JavaScript")
  printf '%s
' "$files" | grep -Eq '\.cs$' && langs+=("C#")
  printf '%s
' "$files" | grep -Eq '\.java$' && langs+=("Java")
  printf '%s
' "$files" | grep -Eq '\.go$' && langs+=("Go")
  printf '%s
' "$files" | grep -Eq '\.py$' && langs+=("Python")

  if [ "${#langs[@]}" -eq 0 ]; then
    return 0
  fi
  join_csv "${langs[@]}"
}

detect_frameworks() {
  local files="$1"
  local frameworks=()

  printf '%s
' "$files" | grep -Eq '\.(tsx|jsx)$' && frameworks+=("React")
  printf '%s
' "$files" | grep -Eiq '(^|/)app/.+/page\.(tsx|jsx|ts|js)$|next\.config' && frameworks+=("Next.js")
  printf '%s
' "$files" | grep -Eq '\.csproj$' && frameworks+=(".NET")
  printf '%s
' "$files" | grep -Eq '^pom\.xml$' && frameworks+=("Spring")
  printf '%s
' "$files" | grep -Eiq '(controller|route|routes)' && frameworks+=("Express")

  if [ "${#frameworks[@]}" -eq 0 ]; then
    return 0
  fi
  join_csv "${frameworks[@]}"
}

detect_package_managers() {
  local files="$1"
  local managers=()

  printf '%s
' "$files" | grep -Eq '^pnpm-lock\.yaml$|^pnpm-workspace\.yaml$' && managers+=("pnpm")
  printf '%s
' "$files" | grep -Eq '^package-lock\.json$' && managers+=("npm")
  printf '%s
' "$files" | grep -Eq '^yarn\.lock$' && managers+=("yarn")
  printf '%s
' "$files" | grep -Eq '\.csproj$' && managers+=("nuget")
  printf '%s
' "$files" | grep -Eq '^pom\.xml$' && managers+=("maven")
  printf '%s
' "$files" | grep -Eq '^go\.mod$' && managers+=("go")

  if [ "${#managers[@]}" -eq 0 ]; then
    return 0
  fi
  join_csv "${managers[@]}"
}

detect_test_tools() {
  local files="$1"
  local tools=()

  printf '%s
' "$files" | grep -Eiq 'vitest|vitest\.config' && tools+=("Vitest")
  printf '%s
' "$files" | grep -Eiq 'jest|jest\.config' && tools+=("Jest")
  printf '%s
' "$files" | grep -Eiq 'playwright|playwright\.config' && tools+=("Playwright")
  printf '%s
' "$files" | grep -Eiq 'cypress|cypress\.config' && tools+=("Cypress")
  printf '%s
' "$files" | grep -Eiq '(\.test\.|\.spec\.)' && tools+=("Existing test files")
  printf '%s
' "$files" | grep -Eiq 'xunit|nunit|mstest' && tools+=(".NET test framework")

  if [ "${#tools[@]}" -eq 0 ]; then
    return 0
  fi
  join_csv "${tools[@]}"
}

infer_contexts() {
  local rel_path="$1"
  local cleaned
  cleaned="$(printf '%s' "$rel_path" | tr '[:upper:]' '[:lower:]' | sed 's#^\./##')"
  cleaned="$(printf '%s' "$cleaned" | awk -F/ '
    {
      count = 0
      for (i = 1; i <= NF; i++) {
        if ($i != "" && $i != "." && $i != "apps" && $i != "services" && $i != "packages" && $i != "libs" && $i != "src" && $i != "workspace" && $i != "projects") {
          parts[++count] = $i
        }
      }
      if (count == 0) {
        print "workspace-root"
      } else if (count == 1) {
        print parts[count]
      } else {
        print parts[count - 1] "," parts[count]
      }
    }'
  )"
  printf '%s' "$cleaned"
}

summarize_repo() {
  local repo_type="$1"
  local contexts="$2"
  local frameworks="$3"

  case "$repo_type" in
    ui) printf 'User-facing application for %s built with %s.' "${contexts:-core workflows}" "${frameworks:-detected frontend tooling}" ;;
    api) printf 'Service or API repository for %s using %s.' "${contexts:-backend capabilities}" "${frameworks:-detected backend tooling}" ;;
    worker) printf 'Background processing repository for %s.' "${contexts:-async workflows}" ;;
    shared-lib) printf 'Shared library or common module for %s.' "${contexts:-cross-cutting concerns}" ;;
    infra) printf 'Infrastructure-as-code repository for provisioning and environment setup.' ;;
    *) printf 'Repository discovered in the workspace for %s.' "${contexts:-project functionality}" ;;
  esac
}

build_route() {
  local file="$1"
  local route
  route="$(printf '%s' "$file" | sed -E 's#^src/pages/##; s#^app/##; s#/page\.(tsx|jsx|ts|js)$##; s#\.(tsx|jsx|ts|js)$##; s#index$##; s#\[([^]]+)\]#:\1#g')"
  route="/$(printf '%s' "$route" | sed 's#//*#/#g; s#/$##')"
  printf '%s' "${route:-/}"
}

GENERATED_AT="$(timestamp_utc)"
REPO_ROOTS="$(collect_repo_roots)"

if [ -z "$REPO_ROOTS" ]; then
  REPO_ROOTS='.'
fi

TEMP_REPOS_FILE="$(mktemp)"
TEMP_SHARED_FILE="$(mktemp)"
TEMP_CONTEXT_FILE="$(mktemp)"
TEMP_FAILINGS_FILE="$(mktemp)"
TEMP_WARNINGS_FILE="$(mktemp)"
trap 'rm -f "$TEMP_REPOS_FILE" "$TEMP_SHARED_FILE" "$TEMP_CONTEXT_FILE" "$TEMP_FAILINGS_FILE" "$TEMP_WARNINGS_FILE"' EXIT

REPO_COUNT=0
SHARED_COUNT=0

while IFS= read -r repo_path; do
  [ -z "$repo_path" ] && continue

  local_rel="$repo_path"
  if [ "$local_rel" = "$ROOT_DIR" ]; then
    local_rel='.'
  fi

  repo_abs="$ROOT_DIR"
  if [ "$local_rel" != "." ]; then
    repo_abs="$local_rel"
  fi

  repo_name="$(basename "$repo_abs")"
  repo_id="repo-$(printf '%s' "$local_rel" | tr '[:upper:]' '[:lower:]' | sed 's#^\./##; s#[^a-z0-9]#-#g; s#--*#-#g; s#^-##; s#-$##')"
  [ -z "$repo_id" ] && repo_id="repo-workspace-root"

  files="$(list_repo_files "$repo_abs")"
  repo_type="$(detect_repo_type "$local_rel" "$files")"
  languages="$(detect_languages "$files")"
  frameworks="$(detect_frameworks "$files")"
  package_managers="$(detect_package_managers "$files")"
  test_tools="$(detect_test_tools "$files")"
  contexts_csv="$(infer_contexts "$local_rel")"
  summary="$(summarize_repo "$repo_type" "${contexts_csv//,/ /}" "$frameworks")"

  context_json=""
  IFS=',' read -r -a context_items <<< "$contexts_csv"
  for context in "${context_items[@]}"; do
    context_trimmed="$(printf '%s' "$context" | sed 's/^ *//; s/ *$//')"
    [ -z "$context_trimmed" ] && continue
    if [ -n "$context_json" ]; then
      context_json="$context_json, "
    fi
    context_json="$context_json"$(json_escape "$context_trimmed")""
  done

  lang_json=""
  IFS=',' read -r -a lang_items <<< "$languages"
  for lang in "${lang_items[@]}"; do
    lang_trimmed="$(printf '%s' "$lang" | sed 's/^ *//; s/ *$//')"
    [ -z "$lang_trimmed" ] && continue
    if [ -n "$lang_json" ]; then
      lang_json="$lang_json, "
    fi
    lang_json="$lang_json"$(json_escape "$lang_trimmed")""
  done

  framework_json=""
  IFS=',' read -r -a framework_items <<< "$frameworks"
  for framework in "${framework_items[@]}"; do
    framework_trimmed="$(printf '%s' "$framework" | sed 's/^ *//; s/ *$//')"
    [ -z "$framework_trimmed" ] && continue
    if [ -n "$framework_json" ]; then
      framework_json="$framework_json, "
    fi
    framework_json="$framework_json"$(json_escape "$framework_trimmed")""
  done

  manager_json=""
  IFS=',' read -r -a manager_items <<< "$package_managers"
  for manager in "${manager_items[@]}"; do
    manager_trimmed="$(printf '%s' "$manager" | sed 's/^ *//; s/ *$//')"
    [ -z "$manager_trimmed" ] && continue
    if [ -n "$manager_json" ]; then
      manager_json="$manager_json, "
    fi
    manager_json="$manager_json"$(json_escape "$manager_trimmed")""
  done

  test_tool_json=""
  IFS=',' read -r -a test_tool_items <<< "$test_tools"
  for test_tool in "${test_tool_items[@]}"; do
    test_tool_trimmed="$(printf '%s' "$test_tool" | sed 's/^ *//; s/ *$//')"
    [ -z "$test_tool_trimmed" ] && continue
    if [ -n "$test_tool_json" ]; then
      test_tool_json="$test_tool_json, "
    fi
    test_tool_json="$test_tool_json"$(json_escape "$test_tool_trimmed")""
  done

  key_files_json=""
  key_files_md=""
  backtick='`'
  while IFS= read -r key_file; do
    [ -z "$key_file" ] && continue
    if [ -n "$key_files_json" ]; then
      key_files_json="$key_files_json, "
      key_files_md="$key_files_md, "
    fi
    key_files_json="$key_files_json"$(json_escape "$key_file")""
    key_files_md="$key_files_md${backtick}$(csv_escape "$key_file")${backtick}"
  done <<EOF
$(printf '%s
' "$files" | grep -Ei '(^|/)(README|routes|router|controller|package\.json|pom\.xml|[^/]+\.csproj)$' | head -n 6)
EOF

  ui_screens_json=""
  while IFS= read -r screen_file; do
    [ -z "$screen_file" ] && continue
    screen_name="$(basename "$screen_file" | sed -E 's/\.(tsx|jsx|ts|js)$//' | sed 's/[-_]/ /g')"
    screen_route="$(build_route "$screen_file")"
    if [ -n "$ui_screens_json" ]; then
      ui_screens_json="$ui_screens_json, "
    fi
    ui_screens_json="$ui_screens_json{ "name": "$(json_escape "$screen_name")", "route": "$(json_escape "$screen_route")" }"
  done <<EOF
$(printf '%s
' "$files" | grep -Ei '(^src/pages/.*\.(tsx|jsx|ts|js)$)|(^app/.+/page\.(tsx|jsx|ts|js)$)' | head -n 8)
EOF

  api_routes_json=""
  while IFS= read -r api_file; do
    [ -z "$api_file" ] && continue
    if [ -n "$api_routes_json" ]; then
      api_routes_json="$api_routes_json, "
    fi
    api_routes_json="$api_routes_json{ "path": "$(json_escape "$api_file")" }"
  done <<EOF
$(printf '%s
' "$files" | grep -Ei '(^|/)(routes|router|controller|handler|api).*(\.(ts|tsx|js|jsx|cs|java|go|py)$)' | head -n 12)
EOF

  [ "$REPO_COUNT" -gt 0 ] && printf ',
' >> "$TEMP_REPOS_FILE"
  cat <<EOF >> "$TEMP_REPOS_FILE"
    {
      "id": "$(json_escape "$repo_id")",
      "name": "$(json_escape "$repo_name")",
      "path": "$(json_escape "$local_rel")",
      "git": {
        "root": "$(json_escape "$local_rel")",
        "branch": null,
        "remote": null
      },
      "status": "discovered",
      "repoType": "$(json_escape "$repo_type")",
      "lifecycle": "active",
      "boundedContexts": [${context_json}],
      "ownership": {
        "team": null,
        "primary": true,
        "confidence": 0.45,
        "source": "workspace-discovery"
      },
      "tech": {
        "languages": [${lang_json}],
        "frameworks": [${framework_json}],
        "packageManagers": [${manager_json}],
        "testTools": [${test_tool_json}]
      },
      "capabilities": {
        "apiRoutes": [${api_routes_json}],
        "uiScreens": [${ui_screens_json}],
        "eventsProduced": [],
        "eventsConsumed": [],
        "datastores": []
      },
      "existingCode": {
        "summary": "$(json_escape "$summary")",
        "keyFiles": [${key_files_json}],
        "knownConstraints": []
      },
      "dependencies": {
        "internalRepos": [],
        "externalServices": []
      },
      "tags": ["$(json_escape "$repo_type")"]
    }
EOF

  if [ "$repo_type" = "shared-lib" ]; then
    [ "$SHARED_COUNT" -gt 0 ] && printf ',
' >> "$TEMP_SHARED_FILE"
    cat <<EOF >> "$TEMP_SHARED_FILE"
    {
      "id": "$(json_escape "$repo_id")",
      "name": "$(json_escape "$repo_name")",
      "path": "$(json_escape "$local_rel")",
      "repoType": "shared-lib",
      "usedBy": [],
      "summary": "$(json_escape "$summary")"
    }
EOF
    SHARED_COUNT=$((SHARED_COUNT + 1))
  fi

  cat <<EOF >> "$TEMP_CONTEXT_FILE"
## $(csv_escape "$repo_name")

- Path: \`$(csv_escape "$local_rel")\`
- Type: \`$(csv_escape "$repo_type")\`
- Bounded contexts: $(csv_escape "${contexts_csv//,/ , }")
- Frameworks: $(csv_escape "${frameworks:-unknown}")
- Package managers: $(csv_escape "${package_managers:-unknown}")
- Test tools: $(csv_escape "${test_tools:-unknown}")
- Key files: ${key_files_md:-_none detected_}
- Summary: $(csv_escape "$summary")

EOF

  REPO_COUNT=$((REPO_COUNT + 1))
done <<EOF
$REPO_ROOTS
EOF

WARNING_JSON=""
if [ "$REPO_COUNT" -le 1 ]; then
  WARNING_JSON='[
      "Only one repository was discovered. Add more repos to the workspace if the feature spans multiple services."
    ]'
else
  WARNING_JSON='[]'
fi

cat <<EOF > "$WORKSPACE_REPOS_FILE"
{
  "version": "1.0",
  "workspaceRoot": "$(json_escape "$ROOT_DIR")",
  "generatedAt": "$(json_escape "$GENERATED_AT")",
  "repos": [
$(cat "$TEMP_REPOS_FILE")
  ],
  "sharedLibraries": [
$(cat "$TEMP_SHARED_FILE")
  ],
  "discovery": {
    "toolVersion": "1.0",
    "unvalidatedRepos": [],
    "missingRepos": [],
    "warnings": $WARNING_JSON
  }
}
EOF

{
printf '# Workspace Context

> Auto-generated from the current cloned workspace.
> Workspace root: `%s`
> Generated: %s

## Repositories (%s)

' "$ROOT_DIR" "$GENERATED_AT" "$REPO_COUNT"
cat "$TEMP_CONTEXT_FILE"
} > "$WORKSPACE_CONTEXT_FILE"

{
printf '# Project Context

> Auto-generated from the current IDE workspace.
> Workspace root: `%s`
> Generated: %s
' "$ROOT_DIR" "$GENERATED_AT"
cat <<'DEVX_PROJECT_CTX_MID'

## Technology Stack And Versions

Review `specs/.devx/workspace-repos.json` for detected languages, frameworks, package managers, and test tools.

## Code Organization Patterns

DEVX_PROJECT_CTX_MID
cat "$TEMP_CONTEXT_FILE"
cat <<'DEVX_PROJECT_CTX_TAIL'

## Critical Implementation Rules

- Read `specs/.devx/current-state.md`, `specs/.devx/implementation-check.md`, and the selected feature change map before coding.
- Verify every suggested file or route in the actual workspace before editing.
- Prefer extending existing modules over creating parallel implementations.
- Existing code patterns override generic or Golden Repo guidance when they conflict.
- Do not require Astra or live Golden Repository access during IDE implementation.
DEVX_PROJECT_CTX_TAIL
} > "$PROJECT_CONTEXT_FILE"

{
printf '# Current State

> Auto-generated from the current IDE workspace.
> Workspace root: `%s`
> Generated: %s

## Repositories And Modules (%s)

' "$ROOT_DIR" "$GENERATED_AT" "$REPO_COUNT"
cat "$TEMP_CONTEXT_FILE"
cat <<'DEVX_CURRENT_STATE_TAIL'

## Existing Code Guidance

- Use `specs/.devx/workspace-repos.json` for machine-readable repo inventory.
- Use `specs/.devx/change-maps/<feature-slug>.md` for feature-specific candidate areas.
- Use the Golden Repo Guidelines file listed in this feature's `prompt.md` (under `specs/.devx/guidance/versions/`) when present, but prefer verified local code patterns on conflict.
DEVX_CURRENT_STATE_TAIL
} > "$CURRENT_STATE_FILE"

append_finding() {
  local target_file="$1"
  local message="$2"
  printf -- '- %s
' "$message" >> "$target_file"
}

FEATURE_DIR_COUNT=0
for feature_dir in "$ROOT_DIR"/specs/*; do
  [ -d "$feature_dir" ] || continue
  feature_slug="$(basename "$feature_dir")"
  [ "$feature_slug" = ".devx" ] && continue
  FEATURE_DIR_COUNT=$((FEATURE_DIR_COUNT + 1))

  if [ ! -f "$feature_dir/specs.md" ]; then
    append_finding "$TEMP_FAILINGS_FILE" "Missing specs file: specs/$feature_slug/specs.md"
  fi
  if [ ! -f "$feature_dir/requirements.md" ]; then
    append_finding "$TEMP_FAILINGS_FILE" "Missing requirements file: specs/$feature_slug/requirements.md"
  fi
  if [ ! -f "$feature_dir/prompt.md" ]; then
    append_finding "$TEMP_FAILINGS_FILE" "Missing IDE prompt file: specs/$feature_slug/prompt.md"
  fi
  if [ ! -f "$CHANGE_MAPS_DIR/$feature_slug.md" ]; then
    append_finding "$TEMP_FAILINGS_FILE" "Missing change map: specs/.devx/change-maps/$feature_slug.md"
  fi
  if [ ! -f "$CHANGE_MAPS_DIR/$feature_slug.json" ]; then
    append_finding "$TEMP_FAILINGS_FILE" "Missing machine-readable change map: specs/.devx/change-maps/$feature_slug.json"
  fi
done

if [ "$FEATURE_DIR_COUNT" -eq 0 ]; then
  append_finding "$TEMP_FAILINGS_FILE" "No feature directories were found under specs/."
fi

GUIDANCE_DIR="$DEVX_DIR/guidance"
# Golden guidance is stored as immutable content-hash versions under
# guidance/versions/<hash>/ (each feature's [G#] ids point at its version).
# There are no flat copies directly under guidance/.
HAS_VERSIONED_GUIDANCE=0
for VERSIONED_GUIDELINES in "$GUIDANCE_DIR"/versions/*/golden-repo-guidelines.md; do
  if [ -s "$VERSIONED_GUIDELINES" ]; then
    HAS_VERSIONED_GUIDANCE=1
    break
  fi
done
if [ "$HAS_VERSIONED_GUIDANCE" -eq 1 ]; then
  : # Versioned guidance present; per-feature [G#] ids resolve against it.
elif [ -d "$GUIDANCE_DIR/versions" ]; then
  append_finding "$TEMP_FAILINGS_FILE" "Missing Golden Repo guidance: no non-empty golden-repo-guidelines.md under specs/.devx/guidance/versions/<hash>/"
else
  append_finding "$TEMP_WARNINGS_FILE" "No local Golden Repo guidance package was found. Continue only if this project has no linked Golden Repo guidance."
fi

CHECK_STATUS="PASS"
CHECK_FINDINGS="- None"
if [ "$REPO_COUNT" -le 0 ]; then
  append_finding "$TEMP_FAILINGS_FILE" "No repositories or modules were discovered."
elif [ "$REPO_COUNT" -le 1 ]; then
  append_finding "$TEMP_WARNINGS_FILE" "Only one repository/module was discovered. Confirm this is expected for this project."
fi

if [ -s "$TEMP_FAILINGS_FILE" ]; then
  CHECK_STATUS="FAIL"
  CHECK_FINDINGS="$(cat "$TEMP_FAILINGS_FILE")"
elif [ -s "$TEMP_WARNINGS_FILE" ]; then
  CHECK_STATUS="CONCERNS"
  CHECK_FINDINGS="$(cat "$TEMP_WARNINGS_FILE")"
fi

{
printf '# Implementation Check

> Auto-generated local readiness gate for IDE code generation.
> Generated: %s

## Status

%s

## Findings

%s

' "$GENERATED_AT" "$CHECK_STATUS" "$CHECK_FINDINGS"
cat <<'DEVX_IMPL_CHECK_TAIL'
## Gate Semantics

- PASS: required local context, feature files, change maps, and any generated guidance package are present.
- CONCERNS: implementation can proceed after human confirmation of the listed warnings.
- FAIL: do not code until the missing local handoff files are regenerated or restored.

## Required Before Coding

- [ ] Read `specs/.devx/project-context.md`.
- [ ] Read `specs/.devx/current-state.md`.
- [ ] Read the selected feature change map under `specs/.devx/change-maps/`.
- [ ] Read local Golden Repo guidance under `specs/.devx/guidance/` when present.
- [ ] Confirm the owning code area by inspecting actual files.

## Rule

Do not start implementation from a Golden Repo link or Astra-only context. Use local files committed or copied with the specs.
DEVX_IMPL_CHECK_TAIL
} > "$IMPLEMENTATION_CHECK_FILE"

if [ ! -f "$REPO_PLANS_FILE" ]; then
  cat <<EOF > "$REPO_PLANS_FILE"
{
  "version": "1.0",
  "generatedAt": "$(json_escape "$GENERATED_AT")",
  "plans": []
}
EOF
fi

echo "Workspace discovery complete. Repositories found: $REPO_COUNT"
echo "Updated specs/.devx/workspace-repos.json, workspace-context.md, project-context.md, current-state.md, and implementation-check.md"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SPECS_DIR/.." && pwd)"
FEATURES_FILE="$SCRIPT_DIR/features.json"
TRACKER_FILE="$SCRIPT_DIR/tracker.json"
GENERATION_FILE="$SCRIPT_DIR/generation.json"
ALLOW_DIRTY="${DEVX_ALLOW_DIRTY:-0}"

if [ "${1:-}" = "--allow-dirty" ]; then
  ALLOW_DIRTY=1
fi

set +e
node - "$FEATURES_FILE" "$TRACKER_FILE" "$GENERATION_FILE" "$SPECS_DIR" <<'NODE'
const fs = require("fs");
const path = require("path");
const [featuresFile, trackerFile, generationFile, specsDir] = process.argv.slice(2);
const allowedStatuses = new Set(["not-started", "in-progress", "done", "blocked", "needs-review", "needs-clarification"]);
const trackerStatuses = new Set(["PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED", "NEEDS_REVIEW", "NEEDS_CLARIFICATION"]);
const issues = [];
const repairs = [];
const absent = [];
const now = new Date().toISOString();

function titleFromSlug(slug) {
  return slug.replace(/[-_]+/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function readJson(file, fallback) {
  if (!fs.existsSync(file)) {
    return fallback;
  }
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function trackerStatusFromFeatureStatus(status) {
  if (status === "done") return "COMPLETED";
  if (status === "in-progress") return "IN_PROGRESS";
  if (status === "blocked") return "BLOCKED";
  if (status === "needs-review") return "NEEDS_REVIEW";
  if (status === "needs-clarification") return "NEEDS_CLARIFICATION";
  return "PENDING";
}

function featureStatusFromTrackerStatus(status) {
  if (status === "COMPLETED") return "done";
  if (status === "IN_PROGRESS") return "in-progress";
  if (status === "BLOCKED") return "blocked";
  if (status === "NEEDS_REVIEW") return "needs-review";
  if (status === "NEEDS_CLARIFICATION") return "needs-clarification";
  return "not-started";
}

const data = readJson(featuresFile, { version: "1.0", generatedAt: now, features: [] });
if (!Array.isArray(data.features)) {
  data.features = [];
  repairs.push("Recreated missing features array.");
}
const tracker = readJson(trackerFile, { trackerVersion: 1, updatedAt: now, tasks: {} });
if (!tracker || typeof tracker !== "object") {
  issues.push("tracker.json is invalid JSON.");
}
if (!tracker.tasks || typeof tracker.tasks !== "object" || Array.isArray(tracker.tasks)) {
  tracker.tasks = {};
  repairs.push("Recreated missing tracker tasks object.");
}
if (typeof tracker.trackerVersion !== "number") {
  tracker.trackerVersion = 1;
  repairs.push("Initialized trackerVersion.");
}
const generation = readJson(generationFile, {
  generationId: "GEN-LOCAL",
  generatedAt: data.generatedAt || now,
  trackerVersion: tracker.trackerVersion,
  taskCount: data.features.length,
});

const specDirs = fs.existsSync(specsDir)
  ? fs.readdirSync(specsDir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && entry.name !== ".devx")
      .map((entry) => entry.name)
  : [];
const dirsWithSpecs = new Set(
  specDirs.filter((slug) => fs.existsSync(path.join(specsDir, slug, "specs.md"))),
);

const seenIds = new Set();
const seenSlugs = new Set();
const seenPaths = new Set();
const nextFeatures = [];
let trackerChanged = false;

for (const feature of data.features) {
  const slug = String(feature.slug || "").trim();
  const id = feature.id;
  if (!slug) {
    issues.push(`Feature "${feature.title || id || "unknown"}" is missing a slug and was skipped.`);
    continue;
  }
  if (seenIds.has(id) || seenSlugs.has(slug)) {
    issues.push(`Duplicate tracking entry removed for slug "${slug}".`);
    repairs.push(`Removed duplicate tracking entry for ${slug}.`);
    continue;
  }
  seenIds.add(id);
  seenSlugs.add(slug);
  const taskId = String(feature.taskId || `FEATURE-${id || slug}`);

  const files = {
    specs: feature.files?.specs || `specs/${slug}/specs.md`,
    requirements: feature.files?.requirements || `specs/${slug}/requirements.md`,
    ...(feature.files?.tddTests ? { tddTests: feature.files.tddTests } : {}),
    prompt: feature.files?.prompt || `specs/${slug}/prompt.md`,
  };

  for (const filePath of Object.values(files)) {
    if (seenPaths.has(filePath)) issues.push(`Duplicate file mapping detected: ${filePath}`);
    seenPaths.add(filePath);
  }

  // A feature can be tracked in features.json/tracker.json without its
  // specs/<slug>/ folder being present in this checkout (features are pushed to
  // the repo selectively). Treat a not-present folder as a non-blocking "skip"
  // rather than a hard error: record it in `absent` (see notice below) instead
  // of `issues`, and leave its tracking entry completely untouched so a later
  // push restores it. Genuine problems (bad JSON, duplicates, missing slug)
  // still go through `issues` and keep blocking.
  const repoRoot = path.dirname(specsDir);
  const featureDir = path.dirname(path.join(repoRoot, files.specs));
  const missingPaths = [files.specs, files.requirements, files.prompt].filter(
    (requiredPath) => !fs.existsSync(path.join(repoRoot, requiredPath)),
  );
  if (missingPaths.length > 0) {
    if (!fs.existsSync(featureDir)) {
      // Whole specs/<slug>/ folder is not in this checkout (features are pushed
      // selectively) -> non-blocking skip; tracking entry left untouched.
      absent.push({ slug, missing: missingPaths });
    } else {
      // Folder is present but a required file is missing -> genuine corruption or
      // a partial push; keep blocking so it is not silently skipped.
      for (const requiredPath of missingPaths) {
        issues.push(`Missing tracked file for "${slug}": ${requiredPath}`);
      }
    }
  }

  const existingTask = tracker.tasks[taskId] || {};
  const taskStatus = trackerStatuses.has(existingTask.status)
    ? existingTask.status
    : trackerStatusFromFeatureStatus(feature.status);
  if (!trackerStatuses.has(existingTask.status)) trackerChanged = true;
  const status = allowedStatuses.has(feature.status)
    ? feature.status
    : featureStatusFromTrackerStatus(taskStatus);
  if (status !== feature.status) repairs.push(`Normalized invalid status for ${slug}.`);
  const nextTask = {
    ...existingTask,
    status: taskStatus,
    title: feature.title,
    featureId: id,
    slug,
    updatedAt: existingTask.updatedAt || data.generatedAt || now,
  };
  if (JSON.stringify(tracker.tasks[taskId]) !== JSON.stringify(nextTask)) {
    tracker.tasks[taskId] = nextTask;
    trackerChanged = true;
  }
  nextFeatures.push({ ...feature, taskId, slug, status, files });
}

const maxId = nextFeatures.reduce((max, feature) => {
  return typeof feature.id === "number" && feature.id > max ? feature.id : max;
}, 0);
let syntheticId = maxId + 1;
for (const slug of dirsWithSpecs) {
  if (seenSlugs.has(slug)) continue;
  const entry = {
    id: syntheticId++,
    title: titleFromSlug(slug),
    slug,
    state: "needs-review",
    status: "needs-review",
    storyCount: 0,
    totalStoryPoints: 0,
    files: {
      specs: `specs/${slug}/specs.md`,
      requirements: `specs/${slug}/requirements.md`,
      prompt: `specs/${slug}/prompt.md`,
    },
    userStories: [],
  };
  const taskId = `FEATURE-${entry.id}`;
  entry.taskId = taskId;
  tracker.tasks[taskId] = {
    status: "NEEDS_REVIEW",
    title: entry.title,
    featureId: entry.id,
    slug,
    updatedAt: now,
  };
  trackerChanged = true;
  nextFeatures.push(entry);
  repairs.push(`Added missing tracking entry for orphan specs folder ${slug}.`);
}

const nextData = {
  ...data,
  generatedAt: data.generatedAt || new Date().toISOString(),
  totalFeatures: nextFeatures.length,
  features: nextFeatures,
};
const nextGeneration = {
  ...generation,
  trackerVersion: tracker.trackerVersion,
  taskCount: nextFeatures.length,
};
if (trackerChanged) {
  tracker.trackerVersion += 1;
  tracker.updatedAt = now;
  nextGeneration.trackerVersion = tracker.trackerVersion;
}
const before = fs.existsSync(featuresFile) ? fs.readFileSync(featuresFile, "utf8") : "";
const after = JSON.stringify(nextData, null, 2) + "\n";
if (before !== after) {
  fs.writeFileSync(featuresFile, after);
}
const trackerBefore = fs.existsSync(trackerFile) ? fs.readFileSync(trackerFile, "utf8") : "";
const trackerAfter = JSON.stringify(tracker, null, 2) + "\n";
if (trackerBefore !== trackerAfter) {
  fs.writeFileSync(trackerFile, trackerAfter);
}
const generationBefore = fs.existsSync(generationFile) ? fs.readFileSync(generationFile, "utf8") : "";
const generationAfter = JSON.stringify(nextGeneration, null, 2) + "\n";
if (generationBefore !== generationAfter) {
  fs.writeFileSync(generationFile, generationAfter);
}

console.log(JSON.stringify({
  ok: issues.length === 0,
  changed: before !== after || trackerBefore !== trackerAfter || generationBefore !== generationAfter,
  issues,
  repairs,
  absent,
}, null, 2));
if (absent.length > 0) {
  console.error("");
  console.error("DevX notice: " + absent.length + " tracked feature folder(s) are not present in this checkout and will be skipped:");
  for (const a of absent) console.error("  - " + a.slug);
  console.error("These entries remain in features.json/tracker.json and will resume automatically once their specs/<slug>/ folder is pushed. This is not an error.");
}
if (issues.length > 0) process.exitCode = 2;
NODE
node_status=$?
set -e
if [ "$node_status" -ne 0 ] && [ "$node_status" -ne 2 ]; then
  exit "$node_status"
fi

if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  upstream="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [ -n "$upstream" ]; then
    git -C "$REPO_ROOT" fetch --quiet || true
    remote_tracker="$(mktemp)"
    if git -C "$REPO_ROOT" show "$upstream:specs/.devx/tracker.json" > "$remote_tracker" 2>/dev/null; then
      set +e
      node - "$TRACKER_FILE" "$remote_tracker" <<'NODE'
const fs = require("fs");
const [localFile, remoteFile] = process.argv.slice(2);
const local = JSON.parse(fs.readFileSync(localFile, "utf8"));
const remote = JSON.parse(fs.readFileSync(remoteFile, "utf8"));
const localTasks = local.tasks || {};
const remoteTasks = remote.tasks || {};
const unsyncedCompleted = [];
const merged = { ...remoteTasks };
let changed = false;

function time(value) {
  const parsed = Date.parse(value || "");
  return Number.isFinite(parsed) ? parsed : 0;
}

for (const [taskId, localTask] of Object.entries(localTasks)) {
  const remoteTask = remoteTasks[taskId];
  const localTime = time(localTask.updatedAt);
  const remoteTime = time(remoteTask && remoteTask.updatedAt);
  if (localTime > remoteTime && localTask.status === "COMPLETED") {
    unsyncedCompleted.push(taskId);
  }
  merged[taskId] = localTime >= remoteTime ? localTask : remoteTask;
  if (JSON.stringify(localTasks[taskId]) !== JSON.stringify(merged[taskId])) changed = true;
}

for (const [taskId, remoteTask] of Object.entries(remoteTasks)) {
  if (!localTasks[taskId]) {
    merged[taskId] = remoteTask;
    changed = true;
  }
}

if (unsyncedCompleted.length > 0) {
  console.error("DevX tracking preflight stopped because completed local tasks are newer than the repository tracker.");
  console.error("Push completed code and specs/.devx/tracker.json before running code generation.");
  console.error("Unsynced completed tasks: " + unsyncedCompleted.join(", "));
  process.exit(5);
}

if (changed || local.trackerVersion !== remote.trackerVersion) {
  const next = {
    ...local,
    trackerVersion: Math.max(Number(local.trackerVersion || 1), Number(remote.trackerVersion || 1)) + 1,
    updatedAt: new Date().toISOString(),
    tasks: merged,
  };
  fs.writeFileSync(localFile, JSON.stringify(next, null, 2) + "\n");
  console.log("Auto-merged specs/.devx/tracker.json with repository tracker state.");
}
NODE
      remote_status=$?
      set -e
      rm -f "$remote_tracker"
      if [ "$remote_status" -ne 0 ]; then
        exit "$remote_status"
      fi
    else
      rm -f "$remote_tracker"
    fi
  fi
  branch_state="$(git -C "$REPO_ROOT" status -sb 2>/dev/null | head -n 1 || true)"
  case "$branch_state" in
    *behind*|*diverged*)
      if [ "$ALLOW_DIRTY" != "1" ]; then
        printf '%s\n' "DevX tracking preflight stopped because this branch is behind or diverged."
        printf '%s\n' "Pull/rebase and resolve tracking updates before running code generation."
        printf '%s\n' "$branch_state"
        exit 4
      fi
      ;;
  esac
  if printf '%s' "$branch_state" | grep -q 'ahead'; then
    if [ "$ALLOW_DIRTY" != "1" ]; then
      printf '%s\n' "DevX tracking preflight stopped because this branch has local commits not pushed to the repository."
      printf '%s\n' "Push completed code and specs/.devx tracking updates before running code generation."
      printf '%s\n' "$branch_state"
      exit 5
    fi
  fi
  dirty="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)"
  blocking_dirty=""
  ignored_setup_dirty=""
  while IFS= read -r dirty_line; do
    [ -z "$dirty_line" ] && continue
    dirty_path="${dirty_line:3}"
    case "$dirty_path" in
      .codex/*|AGENTS.md|specs/.devx/config.json|specs/.devx/workspace-context.md|specs/.devx/features.json|specs/.devx/generation.json|specs/.devx/tracker.json|specs/.devx/logs/*|specs/.devx/tmp/*|specs/.devx/init.lock)
        ignored_setup_dirty="${ignored_setup_dirty}${dirty_line}
"
        ;;
      *)
        blocking_dirty="${blocking_dirty}${dirty_line}
"
        ;;
    esac
  done <<< "$dirty"
  if [ -n "$blocking_dirty" ] && [ "$ALLOW_DIRTY" != "1" ]; then
    printf '%s\n' "DevX tracking preflight stopped because the workspace has uncommitted changes."
    printf '%s\n' "Commit or stash completed code and tracking updates before running code generation."
    printf '%s\n\n' "Use DEVX_ALLOW_DIRTY=1 only when you intentionally want to override this guard."
    if [ -n "$ignored_setup_dirty" ]; then
      printf '%s\n' "Ignored setup-only paths:"
      printf '%s\n' "$ignored_setup_dirty"
      printf '\n'
    fi
    printf '%s\n' "$blocking_dirty"
    exit 3
  fi
fi

if [ "$node_status" -eq 2 ]; then
  printf '%s\n' "DevX tracking preflight repaired what it could, but inconsistencies remain."
  printf '%s\n' "Review specs/.devx/features.json and generated spec folders before continuing."
  exit 2
fi

printf '%s\n' "DevX tracking preflight passed."

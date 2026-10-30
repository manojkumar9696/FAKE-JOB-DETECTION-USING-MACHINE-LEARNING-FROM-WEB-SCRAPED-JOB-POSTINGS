#!/bin/bash
# Astra DevX Init Orchestrator
#
# Usage:
#   bash specs/.devx/init.sh [tool]
#   bash specs/.devx/init.sh --dry-run
#   bash specs/.devx/init.sh --yes codex
#   bash specs/.devx/init.sh --only ai --list-tools

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEVX_DIR="$SCRIPT_DIR"
CONFIG_FILE="$DEVX_DIR/config.json"
WORKSPACE_CONTEXT_FILE="$DEVX_DIR/workspace-context.md"
# Golden stack options live under guidance/versions/<hash>/ (no flat copy).
# Resolve the most recent versioned copy; empty when no golden is present.
GOLDEN_STACK_ENV_FILE="$(ls -1t "$DEVX_DIR"/guidance/versions/*/golden-stack-options.env 2>/dev/null | head -n 1 || true)"
LOCK_FILE="$DEVX_DIR/init.lock"
LOG_DIR="$DEVX_DIR/logs"
TMP_DIR="$DEVX_DIR/tmp"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/init-$RUN_ID.log"

DRY_RUN=0
YES=0
FORCE=0
FORCE_LOCK=0
DEBUG=0
RESET_CONFIG=0
SKIP_VALIDATION=0
ONLY=""
SELECTED_TOOL=""
STEP_RESULTS=""
NON_SCAFFOLDABLE=()
REACT_POST_CREATE="ask"
GOLDEN_UI_OPTIONS=""
GOLDEN_BACKEND_OPTIONS=""
GOLDEN_DATABASE_OPTIONS=""
GOLDEN_AI_OPTIONS=""
GOLDEN_UI_CUSTOM_OPTIONS=""
GOLDEN_BACKEND_CUSTOM_OPTIONS=""
GOLDEN_DATABASE_CUSTOM_OPTIONS=""

AVAILABLE_TOOLS=("claude" "codex" "cursor" "copilot" "windsurf" "cline" "kiro")

if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || printf 0)" -ge 8 ]; then
  C_RESET="$(tput sgr0)"
  C_BOLD="$(tput bold)"
  C_DIM="$(tput dim 2>/dev/null || true)"
  C_BLUE="$(tput setaf 4)"
  C_CYAN="$(tput setaf 6)"
  C_GREEN="$(tput setaf 2)"
  C_YELLOW="$(tput setaf 3)"
  C_RED="$(tput setaf 1)"
  C_MAGENTA="$(tput setaf 5)"
else
  C_RESET=""; C_BOLD=""; C_DIM=""; C_BLUE=""; C_CYAN=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_MAGENTA=""
fi

log_line() {
  local level="$1"
  shift
  local message="$*"
  printf '[%s] [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$level" "$message" >> "$LOG_FILE"
  case "$level" in
    DEBUG) [ "$DEBUG" = "1" ] && printf '%s[debug]%s %s\n' "$C_DIM" "$C_RESET" "$message" ;;
    WARN) printf '%s[warn]%s %s\n' "$C_YELLOW" "$C_RESET" "$message" ;;
    ERROR) printf '%s[error]%s %s\n' "$C_RED" "$C_RESET" "$message" ;;
    SUCCESS) printf '%s[ok]%s %s\n' "$C_GREEN" "$C_RESET" "$message" ;;
    PHASE) printf '\n%s%s==>%s %s\n' "$C_BOLD" "$C_CYAN" "$C_RESET" "$message" ;;
    PROGRESS) printf '%s[*]%s %s\n' "$C_BLUE" "$C_RESET" "$message" ;;
    *) printf '%s\n' "$message" ;;
  esac
  return 0
}

info() { log_line INFO "$@"; }
warn() { log_line WARN "$@"; }
error() { log_line ERROR "$@"; }
debug() { log_line DEBUG "$@"; }
success() { log_line SUCCESS "$@"; }
phase() { log_line PHASE "$@"; }
progress() { log_line PROGRESS "$@"; }

print_banner() {
  printf '\n'
  printf '%s%s\n' "$C_BLUE" '     ___        _______.___________. .______          ___      '
  printf '%s%s\n' "$C_BLUE" '    /   \\      /       |           | |   _  \\        /   \\     '
  printf '%s%s\n' "$C_BLUE" '   /  ^  \\    |   (----`---|  |----` |  |_)  |      /  ^  \\    '
  printf '%s%s\n' "$C_CYAN" '  /  /_\\  \\    \\   \\       |  |      |      /      /  /_\\  \\   '
  printf '%s%s\n' "$C_CYAN" ' /  _____  \\ .----)   |      |  |      |  |\\  \\----./  _____  \\  '
  printf '%s%s%s\n' "$C_CYAN" '/__/     \\__\\|_______/       |__|      | _| `._____/__/     \\__\\ ' "$C_RESET"
  printf '%s%s%s\n' "$C_BOLD$C_CYAN" '             DEVX INIT ORCHESTRATOR' "$C_RESET"
  printf '%s%s%s\n' "$C_DIM" '             Spec-driven workspace bootstrap' "$C_RESET"
  printf '%s%s\n' "$C_CYAN" '----------------------------------------------------------------'
  printf '%sRun id:%s %s\n' "$C_DIM" "$C_RESET" "$RUN_ID"
  printf '%sLog:%s    %s\n\n' "$C_DIM" "$C_RESET" "$LOG_FILE"
  printf '[%s] [INFO] Astra DevX Init Orchestrator\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG_FILE"
  printf '[%s] [INFO] Run id: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" >> "$LOG_FILE"
  printf '[%s] [INFO] Log: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG_FILE" >> "$LOG_FILE"
}

print_kv() {
  local key="$1"
  local value="$2"
  printf '  %s%-18s%s %s\n' "$C_DIM" "$key:" "$C_RESET" "$value"
  printf '[%s] [INFO] %s: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$key" "$value" >> "$LOG_FILE"
}

cleanup_lock() {
  if [ -f "$LOCK_FILE" ] && grep -q "$$" "$LOCK_FILE" 2>/dev/null; then
    rm -f "$LOCK_FILE"
  fi
}

on_interrupt() {
  warn "Init interrupted. Preserving logs at $LOG_FILE"
  if [ "$DRY_RUN" = "0" ]; then
    write_config "cancelled" || true
  fi
  cleanup_lock
  exit 130
}

trap cleanup_lock EXIT
trap on_interrupt INT TERM

usage() {
  cat <<'USAGE'
Astra DevX init

Options:
  --dry-run              Show planned actions without executing scaffold/config writes
  --yes                  Use safe defaults and skip confirmations
  --force                Allow overwriting generated targets after preview
  --force-lock           Replace a stale init lock
  --debug                Print detection and execution debug details
  --only <scope>         Run only ui, backend, database, ai, or discovery
  --skip validation      Skip post-scaffold validation
  --reset-devx-config    Reset specs/.devx/config.json before running
  --list-tools, --list   List AI tools
  -h, --help             Show help

Examples:
  bash specs/.devx/init.sh --dry-run
  bash specs/.devx/init.sh --yes codex
  bash specs/.devx/init.sh --only ai claude
USAGE
}

tool_label() {
  case "$1" in
    claude) printf "Claude Code" ;;
    codex) printf "Codex" ;;
    cursor) printf "Cursor" ;;
    copilot) printf "GitHub Copilot" ;;
    windsurf) printf "Windsurf" ;;
    cline) printf "Cline" ;;
    kiro) printf "Kiro" ;;
    *) printf "%s" "$1" ;;
  esac
}

print_tools() {
  printf '%sAvailable AI tools:%s\n' "$C_BOLD" "$C_RESET"
  local index=1
  for tool in "${AVAILABLE_TOOLS[@]}"; do
    printf '  %s%2s)%s %-18s %s[%s]%s\n' "$C_CYAN" "$index" "$C_RESET" "$(tool_label "$tool")" "$C_DIM" "$tool" "$C_RESET"
    index=$((index + 1))
  done
}

MENU_ACTION=""
MENU_ESCAPE_TAIL=""
read_menu_escape_tail() {
  local rest next
  rest=""
  IFS= read -r -s -n2 rest 2>/dev/null || rest=""
  while [[ "$rest" =~ [0-9\;]$ ]] && IFS= read -r -s -n1 -t 1 next 2>/dev/null; do
    rest="$rest$next"
    case "$next" in [A-Za-z~]) break ;; esac
    [ "${#rest}" -ge 12 ] && break
  done
  MENU_ESCAPE_TAIL="$rest"
}

read_menu_key() {
  local key rest
  MENU_ACTION="other"
  IFS= read -r -s -n1 key 2>/dev/null || key=""
  case "$key" in
    '') MENU_ACTION='enter' ;;
    ' ') MENU_ACTION='enter' ;;
    $'\r'|$'\n') MENU_ACTION='enter' ;;
    $'\x1b')
      read_menu_escape_tail
      rest="$MENU_ESCAPE_TAIL"
      case "$rest" in
        *A) MENU_ACTION='up' ;;
        *B) MENU_ACTION='down' ;;
        *H|*OH|*[1|*[7) MENU_ACTION='home' ;;
        *F|*OF|*[4|*[8) MENU_ACTION='end' ;;
        *) MENU_ACTION='cancel' ;;
      esac
      ;;
    k|K|w|W) MENU_ACTION='up' ;;
    j|J|s|S) MENU_ACTION='down' ;;
    q|Q) MENU_ACTION='cancel' ;;
    [0-9]) MENU_ACTION="number:$key" ;;
    *) MENU_ACTION='other' ;;
  esac
}

MENU_TTY_STATE=""
MENU_PREV_EXIT_TRAP=""
MENU_PREV_INT_TRAP=""
MENU_PREV_TERM_TRAP=""
restore_menu_tty() {
  if [ -n "${MENU_TTY_STATE:-}" ]; then
    stty "$MENU_TTY_STATE" 2>/dev/null || true
    MENU_TTY_STATE=""
  fi
}

restore_menu_traps() {
  if [ -n "$MENU_PREV_EXIT_TRAP" ]; then eval "$MENU_PREV_EXIT_TRAP"; else trap - EXIT; fi
  if [ -n "$MENU_PREV_INT_TRAP" ]; then eval "$MENU_PREV_INT_TRAP"; else trap - INT; fi
  if [ -n "$MENU_PREV_TERM_TRAP" ]; then eval "$MENU_PREV_TERM_TRAP"; else trap - TERM; fi
}

abort_menu() {
  restore_menu_tty
  printf '\n'
  exit 130
}

choose_menu() {
  local title="$1"
  local default_index="$2"
  shift 2
  local options=("$@")
  local count="${#options[@]}"
  local action key starts_at_zero=0 rendered_lines=0
  MENU_CHOICE_INDEX="$default_index"
  [ "$MENU_CHOICE_INDEX" -lt 0 ] && MENU_CHOICE_INDEX=0
  [ "$MENU_CHOICE_INDEX" -ge "$count" ] && MENU_CHOICE_INDEX=$((count - 1))
  [ "$count" -le 0 ] && return 1
  case "${options[0]}" in 0\)*) starts_at_zero=1 ;; esac

  printf '\n%s%s%s\n' "$C_BOLD" "$title" "$C_RESET" >&2
  if [ ! -t 0 ]; then
    local i typed
    for i in "${!options[@]}"; do
      printf '  %s\n' "${options[$i]}" >&2
    done
    printf 'Choice: ' >&2
    if ! read -r typed; then
      typed=""
    fi
    if [ "$starts_at_zero" = "1" ] && [[ "$typed" =~ ^[0-9]+$ ]] && [ "$typed" -ge 0 ] && [ "$typed" -lt "$count" ]; then
      MENU_CHOICE_INDEX="$typed"
    elif [[ "$typed" =~ ^[0-9]+$ ]] && [ "$typed" -ge 1 ] && [ "$typed" -le "$count" ]; then
      MENU_CHOICE_INDEX=$((typed - 1))
    fi
    return 0
  fi

  printf '%sUse Up/Down arrows, j/k, w/s, Home/End, Enter/Space, or press a number. Esc/q cancels.%s\n' "$C_DIM" "$C_RESET" >&2
  MENU_TTY_STATE="$(stty -g 2>/dev/null || true)"
  if [ -n "$MENU_TTY_STATE" ]; then
    MENU_PREV_EXIT_TRAP="$(trap -p EXIT)"
    MENU_PREV_INT_TRAP="$(trap -p INT)"
    MENU_PREV_TERM_TRAP="$(trap -p TERM)"
    stty -echo -icanon min 1 time 0 2>/dev/null || true
    trap restore_menu_tty EXIT
    trap abort_menu INT TERM
  fi
  rendered_lines=$count
  while true; do
    local i
    for i in "${!options[@]}"; do
      if [ "$i" -eq "$MENU_CHOICE_INDEX" ]; then
        printf '  %s> %s%s\n' "$C_CYAN$C_BOLD" "${options[$i]}" "$C_RESET" >&2
      else
        printf '    %s\n' "${options[$i]}" >&2
      fi
    done

    read_menu_key
    action="$MENU_ACTION"
    case "$action" in
      enter) break ;;
      up) MENU_CHOICE_INDEX=$(( (MENU_CHOICE_INDEX + count - 1) % count )) ;;
      down) MENU_CHOICE_INDEX=$(( (MENU_CHOICE_INDEX + 1) % count )) ;;
      home) MENU_CHOICE_INDEX=0 ;;
      end) MENU_CHOICE_INDEX=$((count - 1)) ;;
      cancel)
        MENU_CHOICE_INDEX=-1
        break
        ;;
      number:*)
        key="${action#number:}"
        if [ "$starts_at_zero" = "1" ] && [ "$key" -ge 0 ] && [ "$key" -lt "$count" ]; then
          MENU_CHOICE_INDEX="$key"
          break
        elif [ "$key" -ge 1 ] && [ "$key" -le "$count" ]; then
          MENU_CHOICE_INDEX=$((key - 1))
          break
        fi
        ;;
    esac
    printf '\033[%sA\033[J' "$rendered_lines" >&2
  done
  restore_menu_tty
  restore_menu_traps
  printf '\033[%sA\033[J' "$rendered_lines" >&2
  if [ "$MENU_CHOICE_INDEX" -lt 0 ]; then
    printf '  %sCancelled%s\n' "$C_YELLOW" "$C_RESET" >&2
    warn "Selection cancelled."
    exit 130
  fi
  printf '  %s> %s%s\n' "$C_GREEN$C_BOLD" "${options[$MENU_CHOICE_INDEX]}" "$C_RESET" >&2
}

normalize_tool_choice() {
  local raw
  raw="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/^ *//; s/ *$//; s/[ _-]//g')"
  case "$raw" in
    1|claude|claudecode) printf "claude" ;;
    2|codex|openaicodex) printf "codex" ;;
    3|cursor) printf "cursor" ;;
    4|copilot|githubcopilot|github) printf "copilot" ;;
    5|windsurf) printf "windsurf" ;;
    6|cline) printf "cline" ;;
    7|kiro) printf "kiro" ;;
    *) printf "" ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y) YES=1 ;;
    --force) FORCE=1 ;;
    --force-lock) FORCE_LOCK=1 ;;
    --debug) DEBUG=1 ;;
    --only)
      shift
      ONLY="${1:-}"
      ;;
    --skip)
      shift
      if [ "${1:-}" = "validation" ]; then
        SKIP_VALIDATION=1
      else
        error "Unsupported --skip value: ${1:-}"
        exit 1
      fi
      ;;
    --reset-devx-config) RESET_CONFIG=1 ;;
    --list|--list-tools|-l)
      print_tools
      exit 0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      error "Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      if [ -z "$SELECTED_TOOL" ]; then
        SELECTED_TOOL="$(normalize_tool_choice "$1")"
      else
        warn "Ignoring extra argument: $1"
      fi
      ;;
  esac
  shift
done

if [ -z "$SELECTED_TOOL" ]; then
  SELECTED_TOOL="$(normalize_tool_choice "${AI_TOOL:-}")"
fi

if [ -n "$ONLY" ]; then
  case "$ONLY" in
    ui|backend|database|ai|discovery) ;;
    *) error "--only must be ui, backend, database, ai, or discovery"; exit 1 ;;
  esac
fi

if [ "$DRY_RUN" = "1" ]; then
  LOG_DIR="${TMPDIR:-/tmp}/devx-init-logs"
  TMP_DIR="${TMPDIR:-/tmp}/devx-init-tmp"
  LOCK_FILE="${TMPDIR:-/tmp}/devx-init-$RUN_ID.lock"
  LOG_FILE="$LOG_DIR/init-$RUN_ID.log"
fi

mkdir -p "$LOG_DIR" "$TMP_DIR"

if [ "$RESET_CONFIG" = "1" ] && [ "$DRY_RUN" = "0" ]; then
  rm -f "$CONFIG_FILE"
fi

if [ -f "$LOCK_FILE" ]; then
  if [ "$FORCE_LOCK" = "1" ]; then
    warn "Replacing existing init lock at $LOCK_FILE"
    rm -f "$LOCK_FILE"
  else
    error "Another DevX init appears to be running. Remove $LOCK_FILE or use --force-lock if it is stale."
    exit 1
  fi
fi
printf 'pid=%s\nrunId=%s\nstartedAt=%s\n' "$$" "$RUN_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_FILE"

if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]]; then
  warn "Windows shell detected. This script targets Unix shells via macOS, Linux, Git Bash, or WSL."
fi

print_banner

if [ -z "$SELECTED_TOOL" ]; then
  TOOL_OPTIONS=()
  tool_index=1
  for tool in "${AVAILABLE_TOOLS[@]}"; do
    TOOL_OPTIONS+=("$tool_index) $(tool_label "$tool") [$tool]")
    tool_index=$((tool_index + 1))
  done
  choose_menu "Available AI tools" 0 "${TOOL_OPTIONS[@]}"
  SELECTED_TOOL="${AVAILABLE_TOOLS[$MENU_CHOICE_INDEX]}"
fi

if [ -z "$SELECTED_TOOL" ]; then
  error "Unknown AI tool selection. Run with --list-tools to see supported tools."
  exit 1
fi

success "Selected AI tool: $(tool_label "$SELECTED_TOOL")"

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

safe_version() {
  local command_name="$1"
  if command_exists "$command_name"; then
    "$command_name" --version 2>/dev/null | head -n 1 | tr -d '"'
  else
    printf "missing"
  fi
}

load_golden_stack_options() {
  if [ -f "$GOLDEN_STACK_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$GOLDEN_STACK_ENV_FILE"
    GOLDEN_UI_OPTIONS="$(printf '%s' "${GOLDEN_UI_OPTIONS:-}" | tr -d '\r')"
    GOLDEN_BACKEND_OPTIONS="$(printf '%s' "${GOLDEN_BACKEND_OPTIONS:-}" | tr -d '\r')"
    GOLDEN_DATABASE_OPTIONS="$(printf '%s' "${GOLDEN_DATABASE_OPTIONS:-}" | tr -d '\r')"
    GOLDEN_AI_OPTIONS="$(printf '%s' "${GOLDEN_AI_OPTIONS:-}" | tr -d '\r')"
    GOLDEN_UI_CUSTOM_OPTIONS="$(printf '%s' "${GOLDEN_UI_CUSTOM_OPTIONS:-}" | tr -d '\r')"
    GOLDEN_BACKEND_CUSTOM_OPTIONS="$(printf '%s' "${GOLDEN_BACKEND_CUSTOM_OPTIONS:-}" | tr -d '\r')"
    GOLDEN_DATABASE_CUSTOM_OPTIONS="$(printf '%s' "${GOLDEN_DATABASE_CUSTOM_OPTIONS:-}" | tr -d '\r')"
    debug "Loaded Golden Repo stack options from $GOLDEN_STACK_ENV_FILE"
  fi
}

load_golden_stack_options

first_golden_option() {
  local csv="$1"
  IFS=',' read -r -a GOLDEN_ITEMS <<< "$csv"
  if [ "${#GOLDEN_ITEMS[@]}" -gt 0 ] && [ -n "${GOLDEN_ITEMS[0]}" ]; then
    printf '%s' "${GOLDEN_ITEMS[0]}"
  else
    printf ''
  fi
}

append_csv_options() {
  local csv="$1"
  local value
  IFS=',' read -r -a _tmp_items <<< "$csv"
  for value in "${_tmp_items[@]}"; do
    value="$(printf '%s' "$value" | tr -d '\r' | tr -cd '\11\12\15\40-\176' | sed 's/^ *//; s/ *$//')"
    [ -z "$value" ] && continue
    MENU_LABELS+=("$value")
    MENU_VALUES+=("$value")
  done
}

append_option() {
  local label="$1"
  local value="$2"
  MENU_LABELS+=("$label")
  MENU_VALUES+=("$value")
}

choose_from_labels() {
  local title="$1"
  local default_index="$2"
  local opts=()
  local i=0
  for i in "${!MENU_LABELS[@]}"; do
    opts+=("$i) ${MENU_LABELS[$i]}")
  done
  choose_menu "$title" "$default_index" "${opts[@]}"
  printf '%s' "${MENU_VALUES[$MENU_CHOICE_INDEX]}"
}

prompt_custom_stack_once() {
  local layer_label="$1"
  local typed=""
  while true; do
    printf "Enter custom %s stack: " "$layer_label" >&2
    read -r typed
    typed="$(printf '%s' "$typed" | sed 's/^ *//; s/ *$//')"
    if [ -n "$typed" ]; then
      printf '%s' "$typed"
      return 0
    fi
    warn "Please enter a non-empty value."
  done
}

select_stack_with_golden() {
  local layer_label="$1"
  local extracted_csv="$2"
  local extracted_custom_csv="$3"
  local yes_default="$4"
  local final="none"

  if [ "$YES" = "1" ]; then
    final="$(first_golden_option "$extracted_csv")"
    if [ -z "$final" ]; then
      final="$yes_default"
    fi
    printf '%s' "$final"
    return 0
  fi

  MENU_LABELS=()
  MENU_VALUES=()
  append_csv_options "$extracted_csv"
  append_csv_options "$extracted_custom_csv"

  if [ "${#MENU_LABELS[@]}" -gt 0 ]; then
    append_option "Other" "__OTHER__"
    printf '
%sGolden Repo extracted %s stack options:%s
' "$C_BOLD" "$layer_label" "$C_RESET" >&2
    final="$(choose_from_labels "Choose a $layer_label stack" 0)"
    if [ "$final" != "__OTHER__" ]; then
      printf '%s' "$final"
      return 0
    fi
  fi

  printf '%s' "__OTHER__"
}

detect_package_manager() {
  if [ -f "$REPO_ROOT/pnpm-lock.yaml" ] || [ -f "$REPO_ROOT/pnpm-workspace.yaml" ]; then
    printf "pnpm"
  elif [ -f "$REPO_ROOT/yarn.lock" ]; then
    printf "yarn"
  elif [ -f "$REPO_ROOT/bun.lockb" ] || [ -f "$REPO_ROOT/bun.lock" ]; then
    printf "bun"
  else
    printf "npm"
  fi
}

pm_create() {
  case "$PACKAGE_MANAGER" in
    pnpm) printf "pnpm create" ;;
    yarn) printf "yarn create" ;;
    bun) printf "bun create" ;;
    *) printf "npm create" ;;
  esac
}

pm_install() {
  case "$PACKAGE_MANAGER" in
    pnpm) printf "pnpm install" ;;
    yarn) printf "yarn install" ;;
    bun) printf "bun install" ;;
    *) printf "npm install" ;;
  esac
}

pm_run() {
  case "$PACKAGE_MANAGER" in
    pnpm) printf "pnpm" ;;
    yarn) printf "yarn" ;;
    bun) printf "bun" ;;
    *) printf "npm run" ;;
  esac
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g'
}

score_has_file() {
  local points="$1"
  shift
  local pattern
  for pattern in "$@"; do
    if find "$REPO_ROOT" -maxdepth 4 -path "$REPO_ROOT/specs/.devx" -prune -o -name "$pattern" -print -quit | grep -q .; then
      printf "%s" "$points"
      return
    fi
  done
  printf "0"
}

score_has_path() {
  local points="$1"
  shift
  local path
  for path in "$@"; do
    if [ -e "$REPO_ROOT/$path" ]; then
      printf "%s" "$points"
      return
    fi
  done
  printf "0"
}

score_has_text() {
  local points="$1"
  local pattern="$2"
  local files
  files="$(find "$REPO_ROOT" -maxdepth 4 -name package.json -not -path "*/node_modules/*" -not -path "*/specs/.devx/*" -print 2>/dev/null)"
  if [ -n "$files" ] && printf '%s
' "$files" | xargs grep -E "$pattern" >/dev/null 2>&1; then
    printf "%s" "$points"
  else
    printf "0"
  fi
}

classify_score() {
  local score="$1"
  local conflict="$2"
  if [ "$conflict" = "1" ]; then
    printf "conflict"
  elif [ "$score" -ge 80 ]; then
    printf "detected"
  elif [ "$score" -ge 40 ]; then
    printf "partial"
  else
    printf "missing"
  fi
}

resolve_top_stack() {
  local layer="$1"
  shift
  local best_stack="none"
  local best_score=0
  local second_score=0
  local item stack score
  for item in "$@"; do
    stack="${item%%:*}"
    score="${item##*:}"
    if [ "$score" -gt "$best_score" ]; then
      second_score="$best_score"
      best_score="$score"
      best_stack="$stack"
    elif [ "$score" -gt "$second_score" ]; then
      second_score="$score"
    fi
  done
  local conflict=0
  if [ "$best_score" -ge 40 ] && [ "$second_score" -ge 40 ] && [ $((best_score - second_score)) -le 10 ]; then
    conflict=1
  fi
  printf '%s|%s|%s' "$best_stack" "$best_score" "$(classify_score "$best_score" "$conflict")"
  debug "$layer scores: $* => $best_stack $best_score conflict=$conflict"
}

detect_workspace() {
  phase "[1/6] Detect workspace"
  PACKAGE_MANAGER="$(detect_package_manager)"

  REACT_SCORE=$(( $(score_has_file 35 vite.config.ts vite.config.js) + $(score_has_text 35 '"react"|"@vitejs/plugin-react"') + $(score_has_path 20 src/main.tsx src/App.tsx app/page.tsx) + $(score_has_text 10 '"dev"') ))
  ANGULAR_SCORE=$(( $(score_has_file 50 angular.json) + $(score_has_text 30 '"@angular/core"') + $(score_has_path 20 src/main.ts src/app) ))
  VUE_SCORE=$(( $(score_has_file 35 vite.config.ts vite.config.js vue.config.js) + $(score_has_text 35 '"vue"|"@vitejs/plugin-vue"') + $(score_has_path 20 src/main.ts src/App.vue) ))
  SVELTE_SCORE=$(( $(score_has_file 45 svelte.config.js svelte.config.ts) + $(score_has_text 35 '"svelte"|"@sveltejs/kit"') + $(score_has_path 20 src/routes src/app.html) ))

  NODE_SCORE=$(( $(score_has_text 35 '"express"|"fastify"|"@nestjs/core"') + $(score_has_path 25 server api src/server.ts src/index.ts) + $(score_has_file 20 package.json) + $(score_has_text 20 '"start"|"server"') ))
  PYTHON_SCORE=$(( $(score_has_file 35 pyproject.toml requirements.txt) + $(score_has_text 35 '"fastapi"') + $(score_has_path 20 app/main.py main.py api) ))
  GO_SCORE=$(( $(score_has_file 50 go.mod) + $(score_has_path 25 cmd internal main.go) + $(score_has_file 15 '*.go') ))
  DOTNET_SCORE=$(( $(score_has_file 45 '*.sln' '*.csproj') + $(score_has_path 25 Program.cs Controllers) + $(score_has_text 10 'Microsoft.NET.Sdk.Web') ))
  PHP_SCORE=$(( $(score_has_file 45 composer.json artisan) + $(score_has_path 30 app/Http routes/web.php routes/api.php) ))
  RAILS_SCORE=$(( $(score_has_file 45 Gemfile) + $(score_has_path 30 config/database.yml app/controllers bin/rails) ))

  MYSQL_SCORE=$(( $(score_has_text 30 'mysql|mysql2|MySql') + $(score_has_path 30 migrations prisma schema.prisma database/migrations) ))
  SQLSERVER_SCORE=$(( $(score_has_text 40 'sqlserver|mssql|SqlServer') + $(score_has_path 20 migrations) ))
  POSTGRES_SCORE=$(( $(score_has_text 35 'postgres|pg|postgresql') + $(score_has_path 20 migrations prisma schema.prisma) ))
  MONGO_SCORE=$(( $(score_has_text 40 'mongodb|mongoose|pymongo') + $(score_has_path 20 models schemas) ))
  ORACLE_SCORE=$(( $(score_has_text 45 'oracle|oracledb|Oracle') ))

  IFS='|' read -r UI_STACK UI_SCORE UI_STATUS <<< "$(resolve_top_stack ui "react:$REACT_SCORE" "angular:$ANGULAR_SCORE" "vue:$VUE_SCORE" "svelte:$SVELTE_SCORE")"
  IFS='|' read -r BACKEND_STACK BACKEND_SCORE BACKEND_STATUS <<< "$(resolve_top_stack backend "node:$NODE_SCORE" "python:$PYTHON_SCORE" "go:$GO_SCORE" "dotnet:$DOTNET_SCORE" "php:$PHP_SCORE" "rails:$RAILS_SCORE")"
  IFS='|' read -r DB_STACK DB_SCORE DB_STATUS <<< "$(resolve_top_stack database "mysql:$MYSQL_SCORE" "sqlserver:$SQLSERVER_SCORE" "postgres:$POSTGRES_SCORE" "mongodb:$MONGO_SCORE" "oracle:$ORACLE_SCORE")"

  MONOREPO_MODE="single"
  if [ -f "$REPO_ROOT/pnpm-workspace.yaml" ] || [ -f "$REPO_ROOT/nx.json" ] || [ -f "$REPO_ROOT/turbo.json" ] || [ -f "$REPO_ROOT/lerna.json" ] || [ -d "$REPO_ROOT/apps" ] || [ -d "$REPO_ROOT/packages" ] || [ -d "$REPO_ROOT/services" ]; then
    MONOREPO_MODE="monorepo"
  fi

  NESTED_REPOS="$(find "$REPO_ROOT" -mindepth 2 -maxdepth 4 -name .git -type d -not -path "*/node_modules/*" -not -path "*/specs/.devx/*" | sed "s#^$REPO_ROOT/##" | sed 's#/.git$##' | tr '\n' ',' | sed 's/,$//')"
  CI_MARKERS="$(find "$REPO_ROOT" -maxdepth 4 \( -path "$REPO_ROOT/.github/workflows" -o -name azure-pipelines.yml -o -name .gitlab-ci.yml -o -name Jenkinsfile -o -name docker-compose.yml \) -print 2>/dev/null | sed "s#^$REPO_ROOT/##" | tr '\n' ',' | sed 's/,$//')"
  ENV_MARKERS="$(find "$REPO_ROOT" -maxdepth 4 \( -name ".env" -o -name ".env.example" -o -name "appsettings*.json" \) -not -path "*/node_modules/*" -print 2>/dev/null | sed "s#^$REPO_ROOT/##" | tr '\n' ',' | sed 's/,$//')"

  print_kv "Package manager" "$PACKAGE_MANAGER"
  print_kv "UI" "$UI_STACK ($UI_STATUS, score $UI_SCORE)"
  print_kv "Backend" "$BACKEND_STACK ($BACKEND_STATUS, score $BACKEND_SCORE)"
  print_kv "Database" "$DB_STACK ($DB_STATUS, score $DB_SCORE)"
  print_kv "Workspace mode" "$MONOREPO_MODE"
  [ -n "$NESTED_REPOS" ] && warn "Nested Git repos/submodules detected: $NESTED_REPOS"
  return 0
}

target_for_ui() {
  if [ "$MONOREPO_MODE" = "monorepo" ]; then
    [ -d "$REPO_ROOT/apps" ] && printf "apps/web" || printf "client"
  else
    printf "client"
  fi
}

target_for_backend() {
  if [ "$MONOREPO_MODE" = "monorepo" ]; then
    [ -d "$REPO_ROOT/apps" ] && printf "apps/api" || { [ -d "$REPO_ROOT/services" ] && printf "services/api" || printf "server"; }
  else
    printf "server"
  fi
}

select_missing_layers() {
  phase "[2/6] Decide setup strategy"
  SELECT_UI="none"
  SELECT_BACKEND="none"
  SELECT_DB="none"
  NEEDS_USER_DECISION=""
  local ui_selected=""
  local backend_selected=""
  local db_selected=""

  if { [ -z "$ONLY" ] || [ "$ONLY" = "ui" ]; }; then
    ui_selected="$(select_stack_with_golden "UI" "${GOLDEN_UI_OPTIONS:-}" "${GOLDEN_UI_CUSTOM_OPTIONS:-}" "react")"
    if [ "$ui_selected" != "__OTHER__" ] && [ -n "$ui_selected" ]; then
      SELECT_UI="$ui_selected"
    fi
  fi

  if [ "$SELECT_UI" = "none" ] && [ "$ui_selected" = "__OTHER__" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "ui" ]; }; then
    choose_menu "Choose a UI framework:" 1 \
      "0) None" \
      "1) React (Vite + TypeScript)" \
      "2) Angular (Angular CLI)" \
      "3) Vue (create-vue)" \
      "4) Svelte (sv create)" \
      "5) Other"
    case "$MENU_CHOICE_INDEX" in
      1) SELECT_UI="react" ;;
      2) SELECT_UI="angular" ;;
      3) SELECT_UI="vue" ;;
      4) SELECT_UI="svelte" ;;
      5) SELECT_UI="$(prompt_custom_stack_once "UI")" ;;
      *) SELECT_UI="none" ;;
    esac
  elif [ "$SELECT_UI" = "none" ] && [ "$UI_STATUS" = "missing" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "ui" ]; }; then
    if [ "$YES" = "1" ]; then
      SELECT_UI="react"
    else
      choose_menu "No confident UI setup detected. Choose a UI framework:" 1 \
        "0) None" \
        "1) React (Vite + TypeScript)" \
        "2) Angular (Angular CLI)" \
        "3) Vue (create-vue)" \
        "4) Svelte (sv create)"
      case "$MENU_CHOICE_INDEX" in
        1) SELECT_UI="react" ;;
        2) SELECT_UI="angular" ;;
        3) SELECT_UI="vue" ;;
        4) SELECT_UI="svelte" ;;
        *) SELECT_UI="none" ;;
      esac
    fi
  elif [ "$UI_STATUS" = "conflict" ]; then
    NEEDS_USER_DECISION="$NEEDS_USER_DECISION ui-conflict"
  fi

  if { [ -z "$ONLY" ] || [ "$ONLY" = "backend" ]; }; then
    backend_selected="$(select_stack_with_golden "backend" "${GOLDEN_BACKEND_OPTIONS:-}" "${GOLDEN_BACKEND_CUSTOM_OPTIONS:-}" "node")"
    if [ "$backend_selected" != "__OTHER__" ] && [ -n "$backend_selected" ]; then
      SELECT_BACKEND="$backend_selected"
    fi
  fi

  if [ "$SELECT_BACKEND" = "none" ] && [ "$backend_selected" = "__OTHER__" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "backend" ]; }; then
    choose_menu "Choose a backend stack:" 1 \
      "0) None" \
      "1) Node.js / Express" \
      "2) Python / FastAPI" \
      "3) Go" \
      "4) .NET Web API" \
      "5) PHP / Laravel" \
      "6) Ruby on Rails" \
      "7) Other"
    case "$MENU_CHOICE_INDEX" in
      1) SELECT_BACKEND="node" ;;
      2) SELECT_BACKEND="python" ;;
      3) SELECT_BACKEND="go" ;;
      4) SELECT_BACKEND="dotnet" ;;
      5) SELECT_BACKEND="php" ;;
      6) SELECT_BACKEND="rails" ;;
      7) SELECT_BACKEND="$(prompt_custom_stack_once "backend")" ;;
      *) SELECT_BACKEND="none" ;;
    esac
  elif [ "$SELECT_BACKEND" = "none" ] && [ "$BACKEND_STATUS" = "missing" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "backend" ]; }; then
    if [ "$YES" = "1" ]; then
      SELECT_BACKEND="node"
    else
      choose_menu "No confident backend setup detected. Choose a backend stack:" 1 \
        "0) None" \
        "1) Node.js / Express" \
        "2) Python / FastAPI" \
        "3) Go" \
        "4) .NET Web API" \
        "5) PHP / Laravel" \
        "6) Ruby on Rails"
      case "$MENU_CHOICE_INDEX" in
        1) SELECT_BACKEND="node" ;;
        2) SELECT_BACKEND="python" ;;
        3) SELECT_BACKEND="go" ;;
        4) SELECT_BACKEND="dotnet" ;;
        5) SELECT_BACKEND="php" ;;
        6) SELECT_BACKEND="rails" ;;
        *) SELECT_BACKEND="none" ;;
      esac
    fi
  elif [ "$BACKEND_STATUS" = "conflict" ]; then
    NEEDS_USER_DECISION="$NEEDS_USER_DECISION backend-conflict"
  fi

  if { [ -z "$ONLY" ] || [ "$ONLY" = "database" ]; }; then
    db_selected="$(select_stack_with_golden "database" "${GOLDEN_DATABASE_OPTIONS:-}" "${GOLDEN_DATABASE_CUSTOM_OPTIONS:-}" "none")"
    if [ "$db_selected" != "__OTHER__" ] && [ -n "$db_selected" ]; then
      SELECT_DB="$db_selected"
    fi
  fi

  if [ "$SELECT_DB" = "none" ] && [ "$db_selected" = "__OTHER__" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "database" ]; }; then
    choose_menu "Choose database configuration guidance:" 0 \
      "0) None" \
      "1) MySQL" \
      "2) SQL Server" \
      "3) PostgreSQL" \
      "4) MongoDB" \
      "5) Oracle" \
      "6) Other"
    case "$MENU_CHOICE_INDEX" in
      1) SELECT_DB="mysql" ;;
      2) SELECT_DB="sqlserver" ;;
      3) SELECT_DB="postgres" ;;
      4) SELECT_DB="mongodb" ;;
      5) SELECT_DB="oracle" ;;
      6) SELECT_DB="$(prompt_custom_stack_once "database")" ;;
      *) SELECT_DB="none" ;;
    esac
  elif [ "$SELECT_DB" = "none" ] && [ "$DB_STATUS" = "missing" ] && { [ -z "$ONLY" ] || [ "$ONLY" = "database" ]; }; then
    if [ "$YES" = "1" ]; then
      SELECT_DB="none"
    else
      choose_menu "Choose database configuration guidance:" 0 \
        "0) None" \
        "1) MySQL" \
        "2) SQL Server" \
        "3) PostgreSQL" \
        "4) MongoDB" \
        "5) Oracle"
      case "$MENU_CHOICE_INDEX" in
        1) SELECT_DB="mysql" ;;
        2) SELECT_DB="sqlserver" ;;
        3) SELECT_DB="postgres" ;;
        4) SELECT_DB="mongodb" ;;
        5) SELECT_DB="oracle" ;;
        *) SELECT_DB="none" ;;
      esac
    fi
  fi

  UI_TARGET="$(target_for_ui)"
  BACKEND_TARGET="$(target_for_backend)"

  if [ "$SELECT_UI" = "react" ]; then
    if [ "$YES" = "1" ]; then
      REACT_POST_CREATE="install"
    else
      choose_menu "After creating the React app, what should DevX do?" 1 \
        "0) Create files only" \
        "1) Install dependencies (recommended)" \
        "2) Install and start dev server (long-running)"
      case "$MENU_CHOICE_INDEX" in
        0) REACT_POST_CREATE="none" ;;
        2) REACT_POST_CREATE="install-start" ;;
        *) REACT_POST_CREATE="install" ;;
      esac
    fi
  fi
}

register_steps() {
  STEPS=()
  NON_SCAFFOLDABLE=()
  local ui_step_stack="$SELECT_UI"
  local backend_step_stack="$SELECT_BACKEND"
  case "$ui_step_stack" in react|angular|vue|svelte|nextjs|nuxt|astro|remix|solid|preact|lit|expo|flutter|none) ;; *) ui_step_stack="none" ;; esac
  case "$backend_step_stack" in node|python|go|dotnet|php|rails|nestjs|fastify|django|symfony|phoenix|spring|rust|none) ;; *) backend_step_stack="none" ;; esac
  if [ "$SELECT_UI" != "none" ] && [ "$ui_step_stack" = "none" ]; then NON_SCAFFOLDABLE+=("UI:$SELECT_UI"); fi
  if [ "$SELECT_BACKEND" != "none" ] && [ "$backend_step_stack" = "none" ]; then NON_SCAFFOLDABLE+=("backend:$SELECT_BACKEND"); fi
  if [ -z "$ONLY" ] || [ "$ONLY" = "ui" ]; then
    [ "$ui_step_stack" != "none" ] && STEPS+=("scaffold-ui:$ui_step_stack:$UI_TARGET")
  fi
  if [ -z "$ONLY" ] || [ "$ONLY" = "backend" ]; then
    [ "$backend_step_stack" != "none" ] && STEPS+=("scaffold-backend:$backend_step_stack:$BACKEND_TARGET")
  fi
  if [ -z "$ONLY" ] || [ "$ONLY" = "database" ]; then
    [ "$SELECT_DB" != "none" ] && STEPS+=("configure-database:$SELECT_DB:.")
  fi
  if [ -z "$ONLY" ] || [ "$ONLY" = "discovery" ]; then
    STEPS+=("workspace-discovery:devx:.")
  fi
  if [ -z "$ONLY" ] || [ "$ONLY" = "ai" ]; then
    STEPS+=("configure-ai:$SELECTED_TOOL:.")
  fi
}

step_command() {
  local id="$1"
  local stack="$2"
  local target="$3"
  local create_cmd
  create_cmd="$(pm_create)"
  case "$id:$stack" in
    scaffold-ui:react)
      if [ "${REACT_POST_CREATE:-none}" = "install-start" ]; then
        printf 'printf "y\\n" | %s vite@latest %s -- --template react-ts' "$create_cmd" "$target"
      elif [ "${REACT_POST_CREATE:-none}" = "install" ]; then
        printf 'printf "n\\n" | %s vite@latest %s -- --template react-ts && cd %s && %s' "$create_cmd" "$target" "$target" "$(pm_install)"
      else
        printf 'printf "n\\n" | %s vite@latest %s -- --template react-ts' "$create_cmd" "$target"
      fi
      ;;
    scaffold-ui:angular) printf 'npx -y @angular/cli@latest new %s --routing --style css --skip-git --defaults --skip-install' "$target" ;;
    scaffold-ui:vue) printf '%s vue@latest %s -- --default --typescript --no-git' "$create_cmd" "$target" ;;
    scaffold-ui:svelte) printf 'npx -y sv@latest create %s --template minimal --types ts --no-add-ons --no-install' "$target" ;;
    scaffold-ui:nextjs) printf 'npx -y create-next-app@latest %s --ts --app --eslint --no-tailwind --no-src-dir --import-alias "@/*" --use-npm --skip-install' "$target" ;;
    scaffold-ui:nuxt) printf 'npx -y nuxi@latest init %s --packageManager npm --no-install' "$target" ;;
    scaffold-ui:astro) printf '%s astro@latest %s -- --template minimal --no-install --no-git --skip-houston --typescript strict --yes' "$create_cmd" "$target" ;;
    scaffold-ui:remix) printf 'npx -y create-remix@latest %s --no-install --no-git-init --yes' "$target" ;;
    scaffold-ui:solid) printf 'npx -y degit solidjs/templates/ts %s' "$target" ;;
    scaffold-ui:preact) printf '%s vite@latest %s -- --template preact-ts' "$create_cmd" "$target" ;;
    scaffold-ui:lit) printf '%s vite@latest %s -- --template lit-ts' "$create_cmd" "$target" ;;
    scaffold-ui:expo) printf 'npx -y create-expo-app@latest %s --no-install --yes' "$target" ;;
    scaffold-ui:flutter) printf 'flutter create %s' "$target" ;;
    scaffold-backend:node) printf 'npx -y express-generator@latest %s --no-view --git' "$target" ;;
    scaffold-backend:nestjs) printf 'npx -y @nestjs/cli@latest new %s --skip-git --skip-install --package-manager npm' "$target" ;;
    scaffold-backend:fastify) printf 'mkdir -p %s && npx -y fastify-cli@latest generate %s --lang=ts' "$target" "$target" ;;
    scaffold-backend:django) printf 'mkdir -p %s && cd %s && python3 -m venv .venv && . .venv/bin/activate && python -m pip install django && django-admin startproject app .' "$target" "$target" ;;
    scaffold-backend:symfony) printf 'composer create-project symfony/skeleton %s --no-interaction' "$target" ;;
    scaffold-backend:phoenix) printf 'mix phx.new %s --no-install --no-ecto' "$target" ;;
    scaffold-backend:spring) printf 'curl -fsSL https://start.spring.io/starter.zip -d type=maven-project -d language=java -d dependencies=web -d baseDir=%s -d name=%s -o %s.zip && unzip -o %s.zip && rm -f %s.zip' "$target" "$target" "$target" "$target" "$target" ;;
    scaffold-backend:rust) printf 'cargo new %s --bin' "$target" ;;
    scaffold-backend:python) printf 'mkdir -p %s/app && cd %s && python3 -m venv .venv && . .venv/bin/activate && python -m pip install fastapi uvicorn && printf "from fastapi import FastAPI\\n\\napp = FastAPI()\\n\\n@app.get(\047/health\047)\\ndef health():\\n    return {\047status\047: \047ok\047}\\n" > app/main.py' "$target" "$target" ;;
    scaffold-backend:go) printf 'mkdir -p %s && cd %s && go mod init example.com/devx-api && printf "package main\\n\\nimport (\\n  \042net/http\042\\n)\\n\\nfunc main() {\\n  http.HandleFunc(\042/health\042, func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(\042ok\042)) })\\n  http.ListenAndServe(\042:8080\042, nil)\\n}\\n" > main.go' "$target" "$target" ;;
    scaffold-backend:dotnet) printf 'dotnet new webapi -o %s --no-restore' "$target" ;;
    scaffold-backend:php) printf 'composer create-project laravel/laravel %s' "$target" ;;
    scaffold-backend:rails) printf 'rails new %s --skip-git' "$target" ;;
    configure-database:*) printf 'configure_database %s' "$stack" ;;
    workspace-discovery:*) printf 'bash specs/.devx/discover-workspace.sh' ;;
    configure-ai:*) printf 'configure_ai %s' "$stack" ;;
    *) printf '' ;;
  esac
}

required_tool_for_step() {
  local id="$1"
  local stack="$2"
  case "$id:$stack" in
    scaffold-ui:react|scaffold-ui:vue|scaffold-ui:svelte|scaffold-ui:angular|scaffold-ui:nextjs|scaffold-ui:nuxt|scaffold-ui:astro|scaffold-ui:remix|scaffold-ui:solid|scaffold-ui:preact|scaffold-ui:lit|scaffold-ui:expo|scaffold-backend:node|scaffold-backend:nestjs|scaffold-backend:fastify) printf "node npm" ;;
    scaffold-ui:flutter) printf "flutter" ;;
    scaffold-backend:python|scaffold-backend:django) printf "python3" ;;
    scaffold-backend:go) printf "go" ;;
    scaffold-backend:dotnet) printf "dotnet" ;;
    scaffold-backend:php|scaffold-backend:symfony) printf "php composer" ;;
    scaffold-backend:rails) printf "ruby rails" ;;
    scaffold-backend:phoenix) printf "mix" ;;
    scaffold-backend:spring) printf "curl unzip" ;;
    scaffold-backend:rust) printf "cargo" ;;
    *) printf "" ;;
  esac
}

validate_required_tools() {
  local id="$1"
  local stack="$2"
  local missing=""
  local tool
  for tool in $(required_tool_for_step "$id" "$stack"); do
    if ! command_exists "$tool"; then
      missing="$missing $tool"
    fi
  done
  if [ -n "$missing" ]; then
    warn "Skipping $id/$stack because required tools are missing:$missing"
    return 1
  fi
  return 0
}

preview_plan() {
  phase "[3/6] Preview plan"
  printf '%sPlanned DevX actions%s\n' "$C_BOLD" "$C_RESET"
  print_kv "Workspace" "$MONOREPO_MODE"
  print_kv "Package manager" "$PACKAGE_MANAGER"
  print_kv "Detected UI" "$UI_STACK ($UI_STATUS, score $UI_SCORE)"
  print_kv "Detected backend" "$BACKEND_STACK ($BACKEND_STATUS, score $BACKEND_SCORE)"
  print_kv "Detected database" "$DB_STACK ($DB_STATUS, score $DB_SCORE)"
  [ -n "$NEEDS_USER_DECISION" ] && print_kv "Needs decision" "$NEEDS_USER_DECISION"
  echo ""
  if [ "${#STEPS[@]}" -eq 0 ]; then
    echo "  No scaffold/config steps selected."
  else
    local raw id stack target
    for raw in "${STEPS[@]}"; do
      IFS=':' read -r id stack target <<< "$raw"
      printf '  %s->%s [%s] %s -> %s\n' "$C_CYAN" "$C_RESET" "$id" "$stack" "$target"
      printf '     %s%s%s\n' "$C_DIM" "$(step_command "$id" "$stack" "$target")" "$C_RESET"
    done
  fi
  if [ "${#NON_SCAFFOLDABLE[@]}" -gt 0 ]; then
    echo ""
    printf '%sRecorded but not auto-scaffolded%s\n' "$C_BOLD" "$C_RESET"
    local ns_entry
    for ns_entry in "${NON_SCAFFOLDABLE[@]}"; do
      printf '  %s!%s %s\n' "$C_YELLOW" "$C_RESET" "$ns_entry"
    done
    echo "  init.sh cannot auto-scaffold these stacks (for example KMP, native Android/iOS)."
    echo "  Your selection is saved in specs/.devx/config.json. Ask your AI coding agent to generate the starter project for them."
  fi
  echo ""
  if [ "$DRY_RUN" = "1" ]; then
    info "Dry run requested. No changes will be made."
    return 1
  fi
  if [ "$YES" = "1" ]; then
    return 0
  fi
  printf "Proceed with these actions? [y/N]: "
  read -r confirm
  case "$confirm" in
    y|Y|yes|YES) return 0 ;;
    *) warn "User cancelled execution after preview."; return 1 ;;
  esac
}

mark_step_state() {
  local id="$1"
  local state="$2"
  STEP_RESULTS="$STEP_RESULTS $id=$state"
  debug "step $id => $state"
}

run_with_retry() {
  local command_text="$1"
  local output_log="$2"
  local retry_cleanup_path="${3:-}"
  local attempts=0
  local max_attempts=2
  local command_status=0
  while [ "$attempts" -lt "$max_attempts" ]; do
    attempts=$((attempts + 1))
    if [ "$attempts" -gt 1 ] && [ -n "$retry_cleanup_path" ] && [ -e "$retry_cleanup_path" ]; then
      rm -rf "$retry_cleanup_path"
    fi
    debug "attempt $attempts: $command_text"
    printf '\n--- attempt %s: %s ---\n' "$attempts" "$command_text" >> "$output_log"
    if command_exists timeout; then
      timeout 900 bash -lc "$command_text" >> "$output_log" 2>&1
      command_status=$?
    else
      bash -lc "$command_text" >> "$output_log" 2>&1
      command_status=$?
    fi
    if [ "$command_status" -eq 0 ]; then
      cat "$output_log" >> "$LOG_FILE"
      return 0
    fi
    cat "$output_log" >> "$LOG_FILE"
    warn "Command failed (attempt $attempts/$max_attempts): $command_text"
    warn "See detailed output: $output_log"
  done
  return 1
}

execute_scaffold_step() {
  local id="$1"
  local stack="$2"
  local target="$3"
  local command_text
  local step_log
  local run_dir="$REPO_ROOT"
  local run_target="$target"
  local temp_parent=""
  local target_abs="$REPO_ROOT/$target"
  local target_parent
  local target_name

  if [ "$target" != "." ]; then
    target_parent="$(dirname "$target_abs")"
    target_name="$(basename "$target")"
    temp_parent="$TMP_DIR/scaffold-$RUN_ID-$id-$stack"
    run_dir="$temp_parent"
    run_target="$target_name"
  fi

  command_text="$(step_command "$id" "$stack" "$run_target")"
  step_log="$LOG_DIR/$RUN_ID-$id-$stack.log"

  if ! validate_required_tools "$id" "$stack"; then
    mark_step_state "$id:$stack" "skipped_missing_tools"
    return 0
  fi

  if [ -e "$target_abs" ] && [ "$target" != "." ] && [ "$FORCE" != "1" ]; then
    warn "Target $target already exists. Skipping to avoid overwriting user files."
    mark_step_state "$id:$stack" "skipped_existing_target"
    return 0
  fi

  progress "Executing $id/$stack"
  info "Scaffolding happens in specs/.devx/tmp first, then moves into $target after success."
  info "This can take a minute if packages need to download."
  mark_step_state "$id:$stack" "started"
  if [ -n "$temp_parent" ]; then
    rm -rf "$temp_parent"
    mkdir -p "$temp_parent"
  fi

  if (cd "$run_dir" && run_with_retry "$command_text" "$step_log" "${temp_parent:+$temp_parent/$run_target}"); then
    if [ -n "$temp_parent" ]; then
      if [ ! -d "$temp_parent/$run_target" ]; then
        warn "Scaffold command succeeded, but expected output folder was not created: $temp_parent/$run_target"
        mark_step_state "$id:$stack" "failed_missing_output"
        return 1
      fi
      mkdir -p "$target_parent"
      if [ -e "$target_abs" ]; then
        backup_path="$TMP_DIR/backup-$RUN_ID-$id-$stack-$target_name"
        warn "Backing up existing $target to $backup_path before applying --force output"
        mv "$target_abs" "$backup_path"
      fi
      mv "$temp_parent/$run_target" "$target_abs"
      rm -rf "$temp_parent"
    fi
    mark_step_state "$id:$stack" "succeeded"
    success "$id/$stack completed"
  else
    local failed_dir="$TMP_DIR/failed-$RUN_ID-$id-$stack"
    mkdir -p "$failed_dir"
    if [ -n "$temp_parent" ] && [ -d "$temp_parent" ]; then
      mv "$temp_parent" "$failed_dir/scaffold-output"
    fi
    warn "Step failed. Recovery notes preserved at $failed_dir/README.txt"
    printf 'Failed step: %s/%s\nCommand: %s\nOutput log: %s\nPartial scaffold output: %s/scaffold-output\nRetry after fixing prerequisites.\n' "$id" "$stack" "$command_text" "$step_log" "$failed_dir" > "$failed_dir/README.txt"
    mark_step_state "$id:$stack" "failed"
    return 1
  fi
}

upsert_marked_block() {
  local file="$1"
  local section="$2"
  local body="$3"
  local start="<!-- DEVX:$section:START -->"
  local end="<!-- DEVX:$section:END -->"
  local dir
  dir="$(dirname "$file")"
  mkdir -p "$dir"
  local tmp
  tmp="$(mktemp)"
  if [ -f "$file" ]; then
    awk -v start="$start" -v end="$end" '
      $0 == start { skip=1; next }
      $0 == end { skip=0; next }
      skip != 1 { print }
    ' "$file" > "$tmp"
  fi
  {
    cat "$tmp" 2>/dev/null || true
    printf '\n%s\n%s\n%s\n' "$start" "$body" "$end"
  } > "$file"
  rm -f "$tmp"
}

configure_database() {
  local db="$1"
  local env_file="$REPO_ROOT/.env.example"
  local url
  case "$db" in
    mysql) url="DATABASE_URL=mysql://user:password@localhost:3306/app_db" ;;
    sqlserver) url="DATABASE_URL=sqlserver://user:password@localhost:1433;database=app_db" ;;
    postgres) url="DATABASE_URL=postgresql://user:password@localhost:5432/app_db" ;;
    mongodb) url="DATABASE_URL=mongodb://localhost:27017/app_db" ;;
    oracle) url="DATABASE_URL=oracle://user:password@localhost:1521/app_db" ;;
    *) url="# DATABASE_URL=" ;;
  esac
  upsert_marked_block "$env_file" "ENVIRONMENT" "# DevX example only. Do not store real secrets here.
$url"
  mark_step_state "database:$db" "succeeded"
}

write_workspace_context() {
  info "Updating workspace context"
  local metadata
  metadata="{
  \"schemaVersion\": 1,
  \"runId\": \"$(json_escape "$RUN_ID")\",
  \"workspaceMode\": \"$(json_escape "$MONOREPO_MODE")\",
  \"packageManager\": \"$(json_escape "$PACKAGE_MANAGER")\",
  \"selectedAiTool\": \"$(json_escape "$SELECTED_TOOL")\",
  \"ui\": { \"detected\": \"$(json_escape "$UI_STACK")\", \"status\": \"$(json_escape "$UI_STATUS")\", \"score\": $UI_SCORE, \"selected\": \"$(json_escape "$SELECT_UI")\", \"reactPostCreate\": \"$(json_escape "$REACT_POST_CREATE")\" },
  \"backend\": { \"detected\": \"$(json_escape "$BACKEND_STACK")\", \"status\": \"$(json_escape "$BACKEND_STATUS")\", \"score\": $BACKEND_SCORE, \"selected\": \"$(json_escape "$SELECT_BACKEND")\" },
  \"database\": { \"detected\": \"$(json_escape "$DB_STACK")\", \"status\": \"$(json_escape "$DB_STATUS")\", \"score\": $DB_SCORE, \"selected\": \"$(json_escape "$SELECT_DB")\" }
}"
  cat > "$WORKSPACE_CONTEXT_FILE" <<CONTEXT
# Workspace Context

<!-- DEVX:WORKSPACE-CONTEXT:START -->
Generated by specs/.devx/init.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).

## Workspace Summary

- Mode: $MONOREPO_MODE
- Package manager: $PACKAGE_MANAGER
- AI tool: $(tool_label "$SELECTED_TOOL")
- Nested repositories/submodules: ${NESTED_REPOS:-none}
- CI/CD markers: ${CI_MARKERS:-none}
- Environment markers: ${ENV_MARKERS:-none}

## Detected Architecture

- UI: $UI_STACK ($UI_STATUS, score $UI_SCORE)
- Backend: $BACKEND_STACK ($BACKEND_STATUS, score $BACKEND_SCORE)
- Database: $DB_STACK ($DB_STATUS, score $DB_SCORE)

## Selected Setup

- UI scaffold: $SELECT_UI -> $UI_TARGET
- React post-create action: $REACT_POST_CREATE
- Backend scaffold: $SELECT_BACKEND -> $BACKEND_TARGET
- Database guidance: $SELECT_DB

## Commands

- Install dependencies: $(pm_install)
- Run JS scripts: $(pm_run) <script>
- Workspace discovery: bash specs/.devx/discover-workspace.sh

## Environment And Secrets

Use .env.example for placeholders only. Never commit real secrets.

## Constraints

DevX-managed sections use DEVX markers. User-authored content outside those markers should be preserved.

## Generated Metadata JSON

$metadata
<!-- DEVX:WORKSPACE-CONTEXT:END -->
CONTEXT
}

write_config() {
  local run_status="${1:-completed}"
  cat > "$CONFIG_FILE" <<CONFIG
{
  "schemaVersion": 1,
  "runId": "$(json_escape "$RUN_ID")",
  "status": "$(json_escape "$run_status")",
  "lastRunAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "workspaceMode": "$(json_escape "${MONOREPO_MODE:-unknown}")",
  "packageManager": "$(json_escape "${PACKAGE_MANAGER:-unknown}")",
  "selectedAiTool": "$(json_escape "$SELECTED_TOOL")",
  "only": "$(json_escape "$ONLY")",
  "dryRun": $DRY_RUN,
  "force": $FORCE,
  "skipValidation": $SKIP_VALIDATION,
  "detected": {
    "ui": { "stack": "$(json_escape "${UI_STACK:-none}")", "score": ${UI_SCORE:-0}, "status": "$(json_escape "${UI_STATUS:-unknown}")" },
    "backend": { "stack": "$(json_escape "${BACKEND_STACK:-none}")", "score": ${BACKEND_SCORE:-0}, "status": "$(json_escape "${BACKEND_STATUS:-unknown}")" },
    "database": { "stack": "$(json_escape "${DB_STACK:-none}")", "score": ${DB_SCORE:-0}, "status": "$(json_escape "${DB_STATUS:-unknown}")" }
  },
  "selected": {
    "ui": "$(json_escape "${SELECT_UI:-none}")",
    "backend": "$(json_escape "${SELECT_BACKEND:-none}")",
    "database": "$(json_escape "${SELECT_DB:-none}")",
    "reactPostCreate": "$(json_escape "${REACT_POST_CREATE:-ask}")"
  },
  "targets": {
    "ui": "$(json_escape "${UI_TARGET:-}")",
    "backend": "$(json_escape "${BACKEND_TARGET:-}")"
  },
  "toolVersions": {
    "node": "$(json_escape "$(safe_version node)")",
    "npm": "$(json_escape "$(safe_version npm)")",
    "pnpm": "$(json_escape "$(safe_version pnpm)")",
    "yarn": "$(json_escape "$(safe_version yarn)")",
    "python3": "$(json_escape "$(safe_version python3)")",
    "go": "$(json_escape "$(safe_version go)")",
    "dotnet": "$(json_escape "$(safe_version dotnet)")",
    "php": "$(json_escape "$(safe_version php)")",
    "composer": "$(json_escape "$(safe_version composer)")",
    "ruby": "$(json_escape "$(safe_version ruby)")",
    "rails": "$(json_escape "$(safe_version rails)")"
  },
  "markers": {
    "nestedRepos": "$(json_escape "${NESTED_REPOS:-}")",
    "ci": "$(json_escape "${CI_MARKERS:-}")",
    "environment": "$(json_escape "${ENV_MARKERS:-}")"
  },
  "stepResults": "$(json_escape "${STEP_RESULTS:-}")",
  "needsUserDecision": "$(json_escape "${NEEDS_USER_DECISION:-}")",
  "logFile": "$(json_escape "$LOG_FILE")"
}
CONFIG
}

# Build the universal context from project.md + workflow.md
PROJECT_CONTEXT=$(cat "$DEVX_DIR/project.md")
WORKFLOW_CONTEXT=$(cat "$DEVX_DIR/workflow.md")
CENTRAL_CONTEXT_REFERENCE="This repository uses Astra Spec-Driven Development. Read specs/.devx/workspace-context.md first, then follow specs/.devx/features.json and each specs/<feature-slug>/ folder. Feature presence: a tracked feature is actionable only if specs/<slug>/specs.md exists in this checkout; features are pushed selectively, so a tracked-but-absent folder is expected, not an error. Skip any PENDING feature whose specs/<slug>/specs.md is missing, report the skipped slug, and never modify its features.json/tracker.json entry. Never mark a feature done/COMPLETED unless its specs/<slug>/specs.md and requirements.md were read this session. When specs/<slug>/tasks.md exists, tick each task [x] as its acceptance is met; when a task has a Refer line it cites stable ids (R# in requirements.md, FR# in specs.md Functional Requirements, G# in golden-repo-guidelines.md) that you must open and implement against, treating them as an index into the authoritative detail, while a task with no Refer line is still mandatory and implemented from its Acceptance plus specs.md/requirements.md; if a cited id is missing or clearly mismatched, do not block or follow it blindly - fall back to the task Acceptance plus specs.md/requirements.md and record a one-line note in specs/<slug>/assumptions.md. tasks.md is the ordered guide; requirements.md and the specs.md Acceptance Scenarios are the authoritative gate - never mark the feature done/COMPLETED until every non-blocked task is checked and every requirements.md item and specs.md scenario is satisfied, including any not surfaced as a task. A change-map file is not proof of presence; only specs/<slug>/specs.md is authoritative. Golden Repo guidance: when specs/.devx/guidance/ or specs/.devx/skills/ui-design/ golden files exist, you MUST read them before implementing (required, not optional). The Golden Repo may target a different stack, so do not apply blindly: for each cross-cutting convention (e.g. CSRF synchronizer tokens, i18n/resource files, the UI design system/tokens) either apply it or mark it not-applicable-to-this-stack with a one-line rationale, and include a short Golden Repo Reconciliation note. Verified local patterns override golden guidance on conflict. Do not mark a feature done/COMPLETED when golden artifacts exist unless this read-and-reconcile was completed. Open Questions: before implementing a feature read its Open Questions (features.json openQuestions[] or the specs.md Open Questions section); never leave them write-only. Interactive mode: surface them and ask before building. Unattended/autopilot mode: do not stop to ask - for each Open Question choose a reasonable assumption and record the decision plus a one-line rationale in specs/<slug>/assumptions.md; never assume silently. A feature at status needs-clarification (tracker NEEDS_CLARIFICATION) has a blocking question: skip and report it like an absent folder and leave its entry untouched. Never mark a feature done/COMPLETED while a blocking Open Question is unresolved and unrecorded."

COMBINED_CONTEXT="$PROJECT_CONTEXT

---

$WORKFLOW_CONTEXT"

write_context_file() {
  local target_file="$1"
  local display_name="$2"
  upsert_marked_block "$target_file" "AI-CONTEXT" "$CENTRAL_CONTEXT_REFERENCE"
  info "Updated $display_name"
}

install_mcp_dependencies() {
  local mcp_dir="$DEVX_DIR/mcp"
  echo ""
  info "Setting up MCP server dependencies"
  if [ -d "$mcp_dir" ] && [ -f "$mcp_dir/package.json" ]; then
    if [ "$DRY_RUN" = "1" ]; then
      info "Dry-run: would install MCP dependencies"
    elif command_exists npm; then
      (cd "$mcp_dir" && npm install --silent 2>/dev/null) || warn "MCP dependency install failed; retry manually in $mcp_dir"
      info "MCP server dependencies installed"
    else
      warn "npm is missing; skipping MCP dependency install"
    fi
  else
    warn "MCP server not found at $mcp_dir; skipping"
  fi
}

write_mcp_config() {
  local target_file="$1"
  local display_name="$2"
  local json_body="$3"
  local target_dir
  target_dir="$(dirname "$target_file")"
  mkdir -p "$target_dir"
  if [ "$DRY_RUN" = "1" ]; then
    info "Dry-run: would write $display_name MCP config"
  elif [ ! -f "$target_file" ]; then
    printf '%s
' "$json_body" > "$target_file"
    info "Created $display_name with MCP config"
  else
    warn "$display_name already exists; add MCP config manually if needed"
  fi
}

setup_claude() {
  write_context_file "$REPO_ROOT/CLAUDE.md" "CLAUDE.md"
  install_mcp_dependencies

  MCP_SERVER_PATH="specs/.devx/mcp/server.js"
  write_mcp_config "$REPO_ROOT/.claude/settings.json" ".claude/settings.json" "{
  \"mcpServers\": {
    \"devx-specs\": {
      \"command\": \"node\",
      \"args\": [\"$MCP_SERVER_PATH\"],
      \"cwd\": \".\"
    }
  }
}"

  info "Setting up Claude Code skills"
  SKILLS_SRC="$DEVX_DIR/skills"
  SKILLS_DST="$REPO_ROOT/.claude/skills"
  if [ "$DRY_RUN" = "1" ]; then
    info "Dry-run: would copy Claude skills to .claude/skills"
  elif [ -d "$SKILLS_SRC" ]; then
    mkdir -p "$SKILLS_DST"
    for skill_dir in "$SKILLS_SRC"/*/; do
      if [ -d "$skill_dir" ]; then
        skill_name=$(basename "$skill_dir")
        mkdir -p "$SKILLS_DST/$skill_name"
        cp "$skill_dir"SKILL.md "$SKILLS_DST/$skill_name/" 2>/dev/null || true
      fi
    done
    info "Copied skills to .claude/skills"
  else
    warn "Skills directory not found at $SKILLS_SRC; skipping"
  fi
}

setup_codex_wrappers() {
  local codex_dir="$REPO_ROOT/.codex/devx"
  if [ "$DRY_RUN" = "1" ]; then
    info "Dry-run: would create Codex automation helpers in .codex/devx"
    return
  fi

  mkdir -p "$codex_dir"
  cat > "$codex_dir/implement-next.sh" <<'CODEX_EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEVX_DIR="$REPO_ROOT/specs/.devx"
PROMPT="$(bash "$DEVX_DIR/devx-command.sh" implement-next)"
if command -v codex >/dev/null 2>&1; then
  codex -C "$REPO_ROOT" "$PROMPT" || {
    printf '%s\n\n' "Codex CLI could not start interactively. Use this prompt manually:"
    printf '%s\n' "$PROMPT"
  }
else
  printf '%s\n\n' "Codex CLI was not found. Use this prompt manually:"
  printf '%s\n' "$PROMPT"
fi
CODEX_EOF

  cat > "$codex_dir/implement-feature.sh" <<'CODEX_EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ -z "${1:-}" ]; then
  echo "Usage: bash .codex/devx/implement-feature.sh <feature-slug>" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEVX_DIR="$REPO_ROOT/specs/.devx"
PROMPT="$(bash "$DEVX_DIR/devx-command.sh" implement-feature "$1")"
if command -v codex >/dev/null 2>&1; then
  codex -C "$REPO_ROOT" "$PROMPT" || {
    printf '%s\n\n' "Codex CLI could not start interactively. Use this prompt manually:"
    printf '%s\n' "$PROMPT"
  }
else
  printf '%s\n\n' "Codex CLI was not found. Use this prompt manually:"
  printf '%s\n' "$PROMPT"
fi
CODEX_EOF

  cat > "$codex_dir/validate-feature.sh" <<'CODEX_EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ -z "${1:-}" ]; then
  echo "Usage: bash .codex/devx/validate-feature.sh <feature-slug>" >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEVX_DIR="$REPO_ROOT/specs/.devx"
PROMPT="$(bash "$DEVX_DIR/devx-command.sh" validate-feature "$1")"
if command -v codex >/dev/null 2>&1; then
  codex -C "$REPO_ROOT" "$PROMPT" || {
    printf '%s\n\n' "Codex CLI could not start interactively. Use this prompt manually:"
    printf '%s\n' "$PROMPT"
  }
else
  printf '%s\n\n' "Codex CLI was not found. Use this prompt manually:"
  printf '%s\n' "$PROMPT"
fi
CODEX_EOF

  chmod +x "$codex_dir"/*.sh 2>/dev/null || true
  info "Created Codex automation helpers in .codex/devx"
  echo ""
  echo "Codex automation:"
  echo "  bash .codex/devx/implement-next.sh"
  echo "  bash .codex/devx/implement-feature.sh <feature-slug>"
  echo "  bash .codex/devx/validate-feature.sh <feature-slug>"
}

setup_codex() {
  write_context_file "$REPO_ROOT/AGENTS.md" "AGENTS.md"
  setup_codex_wrappers
}

setup_cursor() {
  write_context_file "$REPO_ROOT/.cursorrules" ".cursorrules"
  install_mcp_dependencies

  MCP_SERVER_PATH="specs/.devx/mcp/server.js"
  write_mcp_config "$REPO_ROOT/.cursor/mcp.json" ".cursor/mcp.json" "{
  \"mcpServers\": {
    \"devx-specs\": {
      \"command\": \"node\",
      \"args\": [\"$MCP_SERVER_PATH\"]
    }
  }
}"
}

setup_copilot() {
  write_context_file "$REPO_ROOT/.github/copilot-instructions.md" ".github/copilot-instructions.md"
}

setup_windsurf() {
  write_context_file "$REPO_ROOT/.windsurfrules" ".windsurfrules"
}

setup_cline() {
  write_context_file "$REPO_ROOT/.clinerules" ".clinerules"
}

setup_kiro() {
  KIRO_DIR="$REPO_ROOT/.kiro/steering"
  mkdir -p "$KIRO_DIR"
  KIRO_FILE="$KIRO_DIR/devx-context.md"
  write_context_file "$KIRO_FILE" ".kiro/steering/devx-context.md"

  install_mcp_dependencies

  MCP_SERVER_PATH="specs/.devx/mcp/server.js"
  write_mcp_config "$REPO_ROOT/.kiro/settings/mcp.json" ".kiro/settings/mcp.json" "{
  \"mcpServers\": {
    \"devx-specs\": {
      \"command\": \"node\",
      \"args\": [\"$MCP_SERVER_PATH\"]
    }
  }
}"
}

configure_ai() {
  local tool="$1"
  case "$tool" in
    claude) setup_claude ;;
    codex) setup_codex ;;
    cursor) setup_cursor ;;
    copilot) setup_copilot ;;
    windsurf) setup_windsurf ;;
    cline) setup_cline ;;
    kiro) setup_kiro ;;
  esac
  mark_step_state "ai:$tool" "succeeded"
}

run_workspace_discovery() {
  DISCOVERY_SCRIPT="$DEVX_DIR/discover-workspace.sh"
  if [ -f "$DISCOVERY_SCRIPT" ]; then
    progress "Running workspace discovery"
    if [ "$DRY_RUN" = "1" ]; then
      info "Dry-run: would run $DISCOVERY_SCRIPT"
    elif bash "$DISCOVERY_SCRIPT"; then
      success "Workspace discovery completed"
    else
      warn "Workspace discovery failed; retry with: bash specs/.devx/discover-workspace.sh"
    fi
  fi
  mark_step_state "discovery" "succeeded"
}

execute_steps() {
  phase "[4/6] Execute selected steps"
  local raw id stack target
  local total index
  total="${#STEPS[@]}"
  index=0
  for raw in "${STEPS[@]}"; do
    index=$((index + 1))
    IFS=':' read -r id stack target <<< "$raw"
    progress "Step $index/$total: $id/$stack"
    case "$id" in
      scaffold-ui|scaffold-backend) execute_scaffold_step "$id" "$stack" "$target" || true ;;
      configure-database) configure_database "$stack" ;;
      workspace-discovery) run_workspace_discovery ;;
      configure-ai)
        write_workspace_context
        configure_ai "$stack"
        ;;
    esac
  done
}

validate_workspace() {
  if [ "$SKIP_VALIDATION" = "1" ] || [ "$DRY_RUN" = "1" ]; then
    info "Validation skipped"
    return
  fi
  phase "[5/6] Validate workspace"
  local run_cmd
  run_cmd="$(pm_run)"
  if [ -f "$REPO_ROOT/package.json" ] && command_exists node; then
    if grep -q '"build"' "$REPO_ROOT/package.json"; then
      (cd "$REPO_ROOT" && $run_cmd build) || warn "Root build validation failed"
    fi
    if grep -q '"test"' "$REPO_ROOT/package.json"; then
      info "Test script detected. Skipping automatic tests by default; run them manually when ready."
    fi
  fi
}

detect_workspace
select_missing_layers
register_steps
if preview_plan; then
  execute_steps
  validate_workspace
  write_workspace_context
  write_config "completed"
else
  if [ "$DRY_RUN" = "0" ]; then
    write_workspace_context
    write_config "planned"
  fi
fi

phase "[6/6] Summary"
if [ "$DRY_RUN" = "1" ]; then
  success "Dry run complete. No workspace files were changed."
else
  success "Done. DevX init status is recorded in specs/.devx/config.json"
fi
if [ "${#NON_SCAFFOLDABLE[@]}" -gt 0 ]; then
  printf '\n%sStacks recorded but not auto-scaffolded%s\n' "$C_BOLD" "$C_RESET"
  for ns_entry in "${NON_SCAFFOLDABLE[@]}"; do
    printf '  %s!%s %s\n' "$C_YELLOW" "$C_RESET" "$ns_entry"
  done
  echo "  init.sh cannot auto-scaffold these (for example KMP, native Android/iOS). Your choice is saved in specs/.devx/config.json."
  echo "  Ask your AI coding agent (Cursor, Copilot, Claude, etc.) to generate the starter project for them, then rerun discovery."
fi
printf '\n%sNext steps%s\n' "$C_BOLD" "$C_RESET"
printf '  %s1.%s Review specs/.devx/workspace-context.md\n' "$C_CYAN" "$C_RESET"
printf '  %s2.%s Review specs/.devx/workspace-repos.json and feature-routing.json\n' "$C_CYAN" "$C_RESET"
printf '  %s3.%s Use your AI tool with the generated central context\n' "$C_CYAN" "$C_RESET"
printf '  %s4.%s If setup was skipped, fix config.json decisions and rerun\n' "$C_CYAN" "$C_RESET"
printf '\n%sLog file%s %s\n' "$C_DIM" "$C_RESET" "$LOG_FILE"
echo ""
case "$SELECTED_TOOL" in
  claude)
    printf '%sClaude Code:%s\n' "$C_BOLD" "$C_RESET"
    echo "  CLAUDE.md references specs/.devx/workspace-context.md. MCP and skills are configured where possible."
    ;;
  codex)
    printf '%sCodex:%s\n' "$C_BOLD" "$C_RESET"
    echo "  AGENTS.md references specs/.devx/workspace-context.md."
    ;;
  cursor)
    printf '%sCursor:%s\n' "$C_BOLD" "$C_RESET"
    echo "  .cursorrules references specs/.devx/workspace-context.md and MCP is configured where possible."
    ;;
  copilot)
    printf '%sGitHub Copilot:%s\n' "$C_BOLD" "$C_RESET"
    echo "  .github/copilot-instructions.md references specs/.devx/workspace-context.md."
    ;;
  windsurf)
    printf '%sWindsurf:%s\n' "$C_BOLD" "$C_RESET"
    echo "  .windsurfrules references specs/.devx/workspace-context.md."
    ;;
  cline)
    printf '%sCline:%s\n' "$C_BOLD" "$C_RESET"
    echo "  .clinerules references specs/.devx/workspace-context.md."
    ;;
  kiro)
    printf '%sKiro:%s\n' "$C_BOLD" "$C_RESET"
    echo "  Steering context references specs/.devx/workspace-context.md and MCP is configured where possible."
    ;;
esac

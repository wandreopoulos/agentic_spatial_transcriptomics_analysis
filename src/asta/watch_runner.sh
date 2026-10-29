#!/usr/bin/env bash
#
# watch_runner.sh — launchd-triggered handler for new GEF drops.
#
# Fired by the com.minjun.sttx-watch LaunchAgent whenever the project folder
# changes. Runs the pipeline once per newly-dropped *.gef, then notifies.
#
# Loop-safety: the pipeline writes outputs back into the watched folder, which
# re-fires this watcher. All watcher state lives OUTSIDE the folder, and we
# silently no-op when (a) no new GEF, (b) the GEF was already processed, or
# (c) a run is already in progress. So re-fires cost ~nothing.

# macOS-only (BSD `stat -f`, ~/Library/Logs) — this script is only used by
# the OpenClaw/launchd workflow. Run it as:
#   asta-watch-launchd [DATA_DIR]     # explicit data folder
#   ST_DATA_DIR=/path asta-watch-launchd   # or via env var
# With neither, it watches the current directory.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: asta-watch-launchd [DATA_DIR]

launchd-triggered handler for new GEF drops (macOS/OpenClaw workflow). Runs
the pipeline once for the newest unprocessed *.gef in DATA_DIR (default:
$ST_DATA_DIR, or the current directory), then notifies. Meant to be invoked
by a launchd agent on a timer, not run interactively.

  DATA_DIR    folder to watch for a .gef file (optional positional arg)
EOF
  exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "asta-watch-launchd is macOS-only (BSD stat, ~/Library/Logs). On Linux use asta-watch; see share/asta/deploy/sttx-watch.service.template." >&2
  exit 1
fi

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # our own install location, to find run_pipeline.sh next to us
PROJECT_DIR="${1:-${ST_DATA_DIR:-$PWD}}"                     # the data folder to watch
STATE_DIR="$HOME/Library/Logs/sttx-watch"
# asta.notify needs the interpreter this package is installed under (it's no
# longer a standalone stdlib-only file), so prefer $ST_PYTHON — set by the
# asta-watch-launchd shim — falling back to whatever python3 is on PATH.
PY="${ST_PYTHON:-$(command -v python3 || echo python3)}"

mkdir -p "$STATE_DIR"
LOCK="$STATE_DIR/run.lock"
PROCESSED="$STATE_DIR/processed.list"
WLOG="$STATE_DIR/watch.log"
touch "$PROCESSED"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WLOG"; }

cd "$PROJECT_DIR" 2>/dev/null || exit 0
PROJECT_DIR="$(pwd)"
# shellcheck disable=SC1091
[[ -f "$PROJECT_DIR/.env" ]] && source "$PROJECT_DIR/.env" 2>/dev/null || true

# --- find newest *.gef (silent exit if none) --------------------------------
shopt -s nullglob
gefs=("$PROJECT_DIR"/*.gef)
shopt -u nullglob
[[ ${#gefs[@]} -eq 0 ]] && exit 0

newest=""; newest_mt=0
for g in "${gefs[@]}"; do
  mt=$(stat -f %m "$g" 2>/dev/null || echo 0)
  if (( mt > newest_mt )); then newest_mt=$mt; newest="$g"; fi
done
[[ -z "$newest" ]] && exit 0

# Identify this drop by path+size; re-dropping a different file reruns.
sig="$newest:$(stat -f %z "$newest" 2>/dev/null || echo 0)"
grep -qxF "$sig" "$PROCESSED" 2>/dev/null && exit 0

# --- atomic lock; silent exit if a run is active (prevents retrigger churn) --
mkdir "$LOCK" 2>/dev/null || exit 0
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# Wait for the file size to stop changing (it may still be copying in).
prev=-1; cur=0
for _ in $(seq 1 60); do
  cur=$(stat -f %z "$newest" 2>/dev/null || echo 0)
  [[ "$cur" == "$prev" && "$cur" != "0" ]] && break
  prev=$cur; sleep 2
done

log "New GEF detected: $newest (size ${cur}B). Starting pipeline."
export ST_INPUT_GEF="$newest"
# Pin the data dir explicitly: don't rely on run_pipeline.sh inheriting our
# $PWD, since a stray $ST_DATA_DIR already in the environment (or set by the
# .env we just sourced) would otherwise win over that and point the pipeline
# at the wrong folder.
export ST_DATA_DIR="$PROJECT_DIR"
start=$(date +%s)
if bash "$SCRIPT_DIR/run_pipeline.sh" >> "$WLOG" 2>&1; then
  status="SUCCESS"
else
  status="FAILED (exit $?)"
fi
dur=$(( $(date +%s) - start ))

# Mark processed (success or fail) so output-write re-fires don't loop.
echo "$sig" >> "$PROCESSED"

summary="Spatial Transcriptomics pipeline ${status} in ${dur}s — file: $(basename "$newest"). Outputs in project folder; details: $WLOG"
log "$summary"
"$PY" -m asta.notify "ST pipeline: ${status}" "$summary" >> "$WLOG" 2>&1 || true

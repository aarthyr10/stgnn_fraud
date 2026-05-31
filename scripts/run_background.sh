#!/usr/bin/env bash
# Run any command sleep-resistant in the background.
#
# Wraps with `caffeinate -i -m -s` so the laptop won't sleep mid-run
# (lid-close is fine; full shutdown is not — for that use the launchd
# plist in scripts/com.aarthy.stgnn.plist).
#
# Logs to artefacts/runs/<timestamp>.log, writes status to
# artefacts/runs/<timestamp>.status.json, and stores the pid for
# inspection.
#
# Usage:
#
#   scripts/run_background.sh python -m scripts.run_pipeline --seed-sweep
#   scripts/run_background.sh python -m scripts.run_pipeline --grid
#
# Tail logs:
#
#   tail -f artefacts/runs/<timestamp>.log
#
# Check status / latest run:
#
#   scripts/run_background.sh status
#   scripts/run_background.sh latest
#
# Kill an in-flight run:
#
#   scripts/run_background.sh stop

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
RUN_DIR="$ROOT/artefacts/runs"
mkdir -p "$RUN_DIR"

PIDFILE="$RUN_DIR/.pid"
LATEST="$RUN_DIR/.latest"

if [[ "${1-}" == "status" ]]; then
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        pid="$(cat "$PIDFILE")"
        latest_log="$(cat "$LATEST" 2>/dev/null || echo "")"
        echo "RUNNING pid=$pid log=$latest_log"
        if [[ -n "$latest_log" && -f "$latest_log" ]]; then
            echo "--- last 20 log lines ---"
            tail -n 20 "$latest_log"
        fi
        exit 0
    fi
    echo "IDLE (no pidfile or stale)"
    if [[ -f "$LATEST" ]]; then
        echo "latest log: $(cat "$LATEST")"
    fi
    exit 0
fi

if [[ "${1-}" == "latest" ]]; then
    if [[ -f "$LATEST" ]]; then
        cat "$LATEST"
    fi
    exit 0
fi

if [[ "${1-}" == "stop" ]]; then
    if [[ -f "$PIDFILE" ]]; then
        pid="$(cat "$PIDFILE")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing pid=$pid"
            # Kill the caffeinate wrapper; the child python dies with it.
            kill -TERM "$pid" || true
            sleep 1
            kill -KILL "$pid" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
    fi
    echo "stopped"
    exit 0
fi

if [[ "${1-}" == "tail" ]]; then
    if [[ -f "$LATEST" ]]; then
        tail -f "$(cat "$LATEST")"
    else
        echo "No prior run."
        exit 1
    fi
fi

if [[ $# -lt 1 ]]; then
    cat <<EOF
Usage: $0 <command...>
       $0 status    # show running job + tail of log
       $0 tail      # tail latest log
       $0 latest    # print path of latest log
       $0 stop      # kill running job

Examples:
  $0 python -m scripts.run_pipeline
  $0 python -m scripts.run_pipeline --force-retrain --note fresh
  $0 python -m scripts.run_pipeline --seed-sweep
  $0 python -m scripts.run_pipeline --grid

The wrapper:
  * Uses 'caffeinate -i -m -s' so lid-close / display-sleep won't pause.
  * Writes logs to $RUN_DIR/<timestamp>.log
  * Writes a status.json with the exit code on completion.
  * Stores pid in $PIDFILE — only one job at a time.
EOF
    exit 1
fi

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "A run is already in progress (pid=$(cat "$PIDFILE"))."
    echo "Use '$0 stop' to cancel it first."
    exit 1
fi

# Resolve python — prefer the project venv if present.
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi
export PATH="$ROOT/.venv/bin:$PATH"

ts="$(date +%Y%m%d_%H%M%S)"
log="$RUN_DIR/${ts}.log"
status="$RUN_DIR/${ts}.status.json"
echo "$log" > "$LATEST"

cmd=("$@")
# If user wrote "python -m scripts...", replace with the resolved python.
if [[ "${cmd[0]}" == "python" || "${cmd[0]}" == "python3" ]]; then
    cmd[0]="$PY"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting: ${cmd[*]}" | tee -a "$log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cwd: $ROOT" | tee -a "$log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] python: $PY" | tee -a "$log"

# Launch under caffeinate; nohup so closing the terminal doesn't kill it;
# disown removes it from the shell's job table.
(
    nohup caffeinate -i -m -s "${cmd[@]}" >> "$log" 2>&1
    rc=$?
    cat > "$status" <<JSON
{
  "started":  "${ts}",
  "finished": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "exit_code": $rc,
  "command":  $(printf '%s\n' "${cmd[@]}" | python3 -c 'import json,sys; print(json.dumps([l.rstrip() for l in sys.stdin]))'),
  "log":      "$log"
}
JSON
    rm -f "$PIDFILE"
    # macOS notification (silent if osascript missing or no GUI).
    if command -v osascript >/dev/null 2>&1; then
        verdict="exit=$rc"
        osascript -e "display notification \"$verdict — see $log\" with title \"stgnn pipeline\"" 2>/dev/null || true
    fi
) &
bgpid=$!
echo "$bgpid" > "$PIDFILE"
disown $bgpid 2>/dev/null || true

echo "Started pid=$bgpid"
echo "Log:    $log"
echo "Status: $status (written on completion)"
echo
echo "Tail with: tail -f $log"
echo "Check:    $0 status"
echo "Stop:     $0 stop"

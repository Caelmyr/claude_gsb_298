#!/usr/bin/env bash
# =============================================================================
# run.sh — one-command launcher for the OR scheduling system.
#
# The project serves BOTH the frontend (HTML pages under frontend/) and the
# backend (Flask REST API) from a single Flask process, so "start frontend +
# backend" is one command.  This script takes care of the environment, seeds
# the example instances, and launches the server.
#
# Usage:
#   ./run.sh                     # start in foreground on 127.0.0.1:5000
#   ./run.sh --port 8080         # custom port
#   ./run.sh --host 0.0.0.0      # listen on all interfaces
#   ./run.sh --background        # start detached (PID file), log to .run/log
#   ./run.sh --stop              # stop a background instance started above
#   ./run.sh --venv              # create/use a local .venv and install deps
#   ./run.sh --no-seed           # do not seed example instances on startup
# =============================================================================

set -euo pipefail

# ---- resolve locations ------------------------------------------------ #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="127.0.0.1"
PORT="5000"
SEED=1
USE_VENV=0
BACKGROUND=0
STOP=0
PID_FILE="$SCRIPT_DIR/.run/server.pid"
LOG_FILE="$SCRIPT_DIR/.run/server.log"

# ---- parse arguments -------------------------------------------------- #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)       HOST="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --background) BACKGROUND=1; shift ;;
    --stop)       STOP=1; shift ;;
    --venv)       USE_VENV=1; shift ;;
    --no-seed)    SEED=0; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---- stop a background instance --------------------------------------- #
if [[ $STOP -eq 1 ]]; then
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID" && echo "stopped background server (pid $PID)"
    else
      echo "no running server (stale pid file)"
    fi
    rm -f "$PID_FILE"
  else
    echo "no background server is running"
  fi
  exit 0
fi

# ---- python environment ----------------------------------------------- #
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found" >&2
  exit 1
fi

if [[ $USE_VENV -eq 1 ]]; then
  if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
    echo "creating virtualenv at .venv ..."
    python3 -m venv "$SCRIPT_DIR/.venv"
  fi
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.venv/bin/activate"
  echo "using virtualenv: $SCRIPT_DIR/.venv"
fi

# ---- dependencies ----------------------------------------------------- #
if ! python3 -c "import flask" >/dev/null 2>&1; then
  echo "flask not found — installing requirements ..."
  if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"
  else
    python3 -m pip install "flask>=3.0"
  fi
fi

# ---- launch ----------------------------------------------------------- #
mkdir -p "$SCRIPT_DIR/.run"
SEED_ARGS=""
[[ $SEED -eq 1 ]] && SEED_ARGS="--seed"

if [[ $BACKGROUND -eq 1 ]]; then
  nohup python3 "$SCRIPT_DIR/run.py" --host "$HOST" --port "$PORT" $SEED_ARGS \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "started in background (pid $(cat "$PID_FILE"))"
  echo "  frontend + api: http://$HOST:$PORT"
  echo "  logs:           $LOG_FILE"
  echo "  stop:           ./run.sh --stop"
else
  echo "starting frontend + backend on http://$HOST:$PORT"
  exec python3 "$SCRIPT_DIR/run.py" --host "$HOST" --port "$PORT" $SEED_ARGS
fi

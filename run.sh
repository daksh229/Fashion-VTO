#!/usr/bin/env bash
# Boot both services for local development.
#
# Starts:
#   api  — FastAPI on http://127.0.0.1:8001
#   web  — Django  on http://127.0.0.1:8000
#
# Output is interleaved in this terminal. Hit Ctrl+C once to stop both.
#
# Note: auto-reload is OFF by default. Reload mode forks worker processes
# that don't always die cleanly on Windows + Git Bash, leaving zombies on
# port 8000/8001. Pass `--reload` if you want it and accept the trade-off.

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cross-platform venv python lookup. Windows ships venv as Scripts\python.exe;
# Linux/macOS use bin/python. Try Windows first since this repo is developed there.
VENV_PY="$PROJECT_ROOT/venv/Scripts/python.exe"
if [[ ! -x "$VENV_PY" ]]; then
  VENV_PY="$PROJECT_ROOT/venv/bin/python"
fi
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: venv python not found. Expected at:"
  echo "  $PROJECT_ROOT/venv/Scripts/python.exe (Windows)"
  echo "  $PROJECT_ROOT/venv/bin/python         (Linux/macOS)"
  exit 1
fi

# Silence MediaPipe's native clearcut telemetry retries.
export GLOG_minloglevel=3

UVICORN_FLAGS=()
DJANGO_FLAGS=("--noreload")
if [[ "${1:-}" == "--reload" ]]; then
  UVICORN_FLAGS+=("--reload")
  DJANGO_FLAGS=()
  echo "[run] --reload requested; orphan processes possible on Windows after Ctrl+C"
fi

cleanup() {
  echo
  echo "[run] shutting down..."
  # First try: regular bash kill on tracked PIDs. Works cleanly on Linux/macOS.
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "${WEB_PID:-}" ]] && kill "$WEB_PID" 2>/dev/null || true
  # Windows + Git Bash fallback: bash PIDs are MSYS pseudo-PIDs that don't
  # match Windows PIDs, so the kills above don't always reach python.exe.
  # Look up whatever is currently bound to our ports (the Windows PID) and
  # tree-kill it. No-op on platforms without taskkill.
  if command -v taskkill.exe >/dev/null 2>&1; then
    sleep 1
    for port in 8001 8000; do
      pid=$(netstat -ano 2>/dev/null \
        | awk -v p=":$port" '$2 ~ p"$" && $4=="LISTENING" {print $5; exit}')
      if [[ -n "$pid" ]]; then
        taskkill.exe //F //T //PID "$pid" >/dev/null 2>&1 || true
      fi
    done
  fi
  wait 2>/dev/null || true
  echo "[run] done."
}
trap cleanup INT TERM EXIT

echo "[run] api  → http://127.0.0.1:8001"
"$VENV_PY" -m uvicorn api.main:app --host 127.0.0.1 --port 8001 "${UVICORN_FLAGS[@]}" &
API_PID=$!

echo "[run] web  → http://127.0.0.1:8000"
( cd "$PROJECT_ROOT/web" && exec "$VENV_PY" manage.py runserver 127.0.0.1:8000 "${DJANGO_FLAGS[@]}" ) &
WEB_PID=$!

echo "[run] both started — open http://127.0.0.1:8000/ in a browser"
echo "[run] Ctrl+C to stop"
wait

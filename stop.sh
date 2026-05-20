#!/usr/bin/env bash
# Stop both services by port. Use this if `run.sh` left zombies behind
# (common on Windows + Git Bash, where bash signal handling can't always
# propagate Ctrl+C to spawned python.exe processes).

set -u

API_PORT=8001
WEB_PORT=8000

stopped_any=0

if command -v taskkill.exe >/dev/null 2>&1; then
  # Windows path: find the Windows PID listening on each port and tree-kill it.
  for port in "$API_PORT" "$WEB_PORT"; do
    pid=$(netstat -ano 2>/dev/null \
      | awk -v p=":$port" '$2 ~ p"$" && $4=="LISTENING" {print $5; exit}')
    if [[ -n "$pid" ]]; then
      echo "[stop] killing PID $pid (port $port)"
      taskkill.exe //F //T //PID "$pid" >/dev/null 2>&1 && stopped_any=1
    fi
  done
elif command -v lsof >/dev/null 2>&1; then
  # Linux/macOS path: lsof gives us the listener PID directly.
  for port in "$API_PORT" "$WEB_PORT"; do
    pid=$(lsof -ti tcp:"$port" -s tcp:LISTEN 2>/dev/null | head -1)
    if [[ -n "$pid" ]]; then
      echo "[stop] killing PID $pid (port $port)"
      kill "$pid" 2>/dev/null && stopped_any=1
    fi
  done
else
  echo "[stop] neither taskkill nor lsof found — kill manually."
  exit 1
fi

if [[ $stopped_any -eq 0 ]]; then
  echo "[stop] nothing was listening on $API_PORT or $WEB_PORT."
fi

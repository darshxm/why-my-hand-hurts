#!/usr/bin/env bash
set -euo pipefail

# Navigate to repo root (directory of this script)
cd "$(dirname "$0")"

# Ensure virtualenv exists
if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Run ./setup.sh first."
  exit 1
fi

# Activate venv
source .venv/bin/activate

# Ensure project root on PYTHONPATH for dashboard/server.py
export PYTHONPATH="${PYTHONPATH:-.}"

# Free the port if already in use
PORT=${PORT:-5000}
if lsof -i tcp:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT in use; killing existing listeners..."
  lsof -ti tcp:$PORT -sTCP:LISTEN | xargs -r kill
  sleep 0.5
fi

# Start the Flask API in the background
python dashboard/server.py keylog.csv --host 127.0.0.1 --port "$PORT" &
API_PID=$!

echo "Dashboard started on http://127.0.0.1:${PORT}"
echo "Open that URL in your browser to view the charts."
echo "Press Ctrl+C to stop."

# Wait on background process
wait ${API_PID}

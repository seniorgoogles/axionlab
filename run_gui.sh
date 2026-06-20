#!/usr/bin/env bash
# Launch the axionlab control web app (FastAPI + uvicorn).
#
#   ./run_gui.sh                 # serve on http://127.0.0.1:8000
#   ./run_gui.sh --port 8123     # pick another port
#   HOST=0.0.0.0 ./run_gui.sh    # expose on the network
#   ./run_gui.sh --reload        # auto-reload on code changes (dev)
#
# Extra args are passed straight through to uvicorn.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"   # repo root

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# the GUI itself only needs fastapi/uvicorn/pyyaml; the jobs it launches
# (train/qat/export/finn) pull the heavy deps lazily via `python run.py ...`.
if ! python -c "import fastapi, uvicorn, yaml" 2>/dev/null; then
    echo "Missing GUI deps. Install with:" >&2
    echo "    pip install fastapi uvicorn pyyaml   # GUI only" >&2
    echo "    pip install -r requirements.txt      # full toolchain" >&2
    exit 1
fi

echo "axionlab control -> http://${HOST}:${PORT}"
exec uvicorn src.webapp.server:app --host "$HOST" --port "$PORT" "$@"

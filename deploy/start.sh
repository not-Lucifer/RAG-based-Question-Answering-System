#!/usr/bin/env bash
# Start the API and the UI in one container (free-tier hosting).
set -euo pipefail
cd "$(dirname "$0")/.."

uvicorn app.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Seed the sample notes once the API is up; runs in the background so the UI appears at once.
python scripts/seed_library.py --path data/samples --wait 120 &

streamlit run frontend/streamlit_app.py \
  --server.port "${PORT:-8501}" --server.address 0.0.0.0 --server.headless true \
  --server.enableXsrfProtection false --server.enableCORS false &
UI_PID=$!

# If either process dies, stop the container so the platform restarts it.
wait -n "$API_PID" "$UI_PID"
exit $?

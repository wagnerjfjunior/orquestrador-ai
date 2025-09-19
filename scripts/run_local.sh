#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e . uvicorn[standard]
set -a; source .env; set +a
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

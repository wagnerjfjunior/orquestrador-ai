#!/usr/bin/env bash
set -euo pipefail
set -a; source .env; set +a
docker compose up -d --build
docker compose ps

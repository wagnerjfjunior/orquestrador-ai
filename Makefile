SHELL := /bin/bash

.PHONY: help up down logs rebuild restart lint env-check run-local

help:
	@echo "Targets:"
	@echo "  make up        - Build & start container (docker compose)"
	@echo "  make down      - Stop container"
	@echo "  make logs      - Tail logs"
	@echo "  make rebuild   - Force rebuild image and start"
	@echo "  make restart   - Restart service"
	@echo "  make env-check - Validate .env keys are present"
	@echo "  make run-local - Run uvicorn locally on :8001"

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

rebuild:
	docker compose build --no-cache
	docker compose up -d

restart:
	docker compose restart

env-check:
	@python scripts/env_check.py

run-local:
	@source .venv/bin/activate && \
	 uvicorn app.main:app --reload --host 0.0.0.0 --port 8001


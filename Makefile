PYTHON ?= python
PIP ?= pip

.PHONY: install
install:
	$(PYTHON) -m pip install --upgrade pip
	if [ -f requirements.txt ]; then $(PIP) install -r requirements.txt; fi
	$(PIP) install -e .
	$(PIP) install pytest ruff flake8 "uvicorn[standard]"

.PHONY: lint
lint:
	ruff check .
	flake8 .

.PHONY: fmt
fmt:
	ruff check . --fix

.PHONY: test
test:
	pytest -q

.PHONY: run
run:
	uvicorn app.main:app --host 0.0.0.0 --port $${PORT:-8000}

.PHONY: clean
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + || true
	find . -name "*.pyc" -type f -delete || true
	rm -rf .pytest_cache .mypy_cache .pytype dist build *.egg-info || true

IMAGE ?= orquestrador-ai:latest

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE) .

.PHONY: docker-run
docker-run:
	docker run --rm -p 8000:8000 -e PORT=8000 $(IMAGE)

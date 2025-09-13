FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip install -e . && \
    pip install "uvicorn[standard]" gunicorn

EXPOSE 8000

ENV LOG_LEVEL=INFO \
    OPENAI_API_KEY="" \
    GEMINI_API_KEY="" \
    APP_VERSION="docker"

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

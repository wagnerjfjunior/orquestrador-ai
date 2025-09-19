# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_VERSION=2025-09-18+strategy \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    # Modelos com valores default (NÃO inclui chaves!)
    GEMINI_MODEL=gemini-2.0-flash \
    OPENAI_MODEL=gpt-4o-mini

# deps do sistema (curl p/ healthcheck e build-base mínimo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala as dependências do projeto
COPY pyproject.toml /app/pyproject.toml
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -e .

# Copia apenas o código da app
COPY app /app/app

EXPOSE 8000

# Healthcheck dentro do container
HEALTHCHECK --interval=20s --timeout=5s --retries=10 --start-period=10s \
  CMD curl -fsS http://localhost:8000/health || exit 1

# roda uvicorn simples; sem --reload em produção
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

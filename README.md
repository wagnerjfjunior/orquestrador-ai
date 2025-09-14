# orquestrador-ai

Orquestrador multi-IA (OpenAI / Gemini / Echo) com:
- **FastAPI**
- **Observabilidade**: logs estruturados (structlog) + métricas Prometheus em `/metrics`
- **Fallback** entre provedores em `provider=auto`
- **X-Request-ID** gerado/propagado em todas as respostas
- **Testes** (pytest) e **CI** (GitHub Actions)
- **Dockerfile** e **Makefile** para ciclo local/CI

---

## Sumário
- [Rotas](#rotas)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Rodar local (venv)](#rodar-local-venv)
- [Makefile (atalhos)](#makefile-atalhos)
- [Docker (local)](#docker-local)
- [Métricas (Prometheus)](#métricas-prometheus)
- [Logs estruturados](#logs-estruturados)
- [CI (GitHub Actions)](#ci-github-actions)
- [Deploy (Render)](#deploy-render)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Testes](#testes)
- [Troubleshooting](#troubleshooting)

---

## Rotas

- `GET /` → `{"status":"live"}`
- `GET /health` → `{"status":"ok"}`  
  - **HEAD /health** também é aceito (evita 405 em clientes que só fazem HEAD)
- `GET /ready` → `{"status":"ready"}`
- `POST /ask?provider=echo|openai|gemini|auto&use_fallback=true`
  ```json
  { "prompt": "olá" }

# Orquestrador AI

Orquestrador multi-IA (OpenAI / Gemini / Echo) com **FastAPI**, observabilidade e métricas **Prometheus**.  
Inclui logs estruturados, fallback automático entre provedores e testes de integração.

---

## 🚀 Rotas

- `GET /` → `{"status":"live"}`
- `GET /health` → `{"status":"ok"}` (suporta também `HEAD /health`)
- `GET /ready` → `{"status":"ready"}`
- `POST /ask?provider=echo|openai|gemini|auto&use_fallback=true`
  ```json
  { "prompt": "olá" }




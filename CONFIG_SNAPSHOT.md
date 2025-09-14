# CONFIG SNAPSHOT

- Generated at: **2025-09-14 17:22:42 **

- Root: `/Users/wagnerjfjunior/orquestrador-ai`

- Files: **36**

- Total config lines: **2134**


---

## Index

- [ 1] `.env.example` — 29 lines — mtime 2025-09-13 07:55:28 — sha256 `274aceefe36c…`
- [ 2] `.github/workflows/ci.yml` — 39 lines — mtime 2025-09-13 19:22:11 — sha256 `1741b400f725…`
- [ 3] `.github/workflows/snapshot.yml` — 21 lines — mtime 2025-09-14 12:15:11 — sha256 `73f13efc7425…`
- [ 4] `.gitignore` — 10 lines — mtime 2025-09-12 01:56:56 — sha256 `b746016476e6…`
- [ 5] `app/__init__.py` — 2 lines — mtime 2025-09-13 15:30:17 — sha256 `f0fb5e1d3cbe…`
- [ 6] `app/cache.py` — 34 lines — mtime 2025-09-13 18:33:31 — sha256 `0a8edfe8a856…`
- [ 7] `app/config.py` — 62 lines — mtime 2025-09-13 18:33:31 — sha256 `ffa9c871ab06…`
- [ 8] `app/gemini_client.py` — 83 lines — mtime 2025-09-14 16:13:34 — sha256 `a75beba70841…`
- [ 9] `app/judge.py` — 71 lines — mtime 2025-09-14 11:27:38 — sha256 `874ade3ba271…`
- [10] `app/main.py` — 205 lines — mtime 2025-09-14 16:18:50 — sha256 `560ecf5cacee…`
- [11] `app/metrics.py` — 44 lines — mtime 2025-09-13 18:33:31 — sha256 `672d96010d9c…`
- [12] `app/observability.py` — 111 lines — mtime 2025-09-14 11:32:26 — sha256 `5a5528e1c8ab…`
- [13] `app/openai_client.py` — 114 lines — mtime 2025-09-14 16:00:55 — sha256 `8526780b124e…`
- [14] `app/retry.py` — 42 lines — mtime 2025-09-13 18:33:31 — sha256 `f7539c2a2673…`
- [15] `app/utils/__init__.py` — 2 lines — mtime 2025-09-13 15:31:19 — sha256 `f0fb5e1d3cbe…`
- [16] `app/utils/retry.py` — 42 lines — mtime 2025-09-13 18:33:31 — sha256 `d22081dab42b…`
- [17] `CONFIG_SNAPSHOT.manifest.json` — 225 lines — mtime 2025-09-14 17:21:00 — sha256 `48ae3dd7a70c…`
- [18] `cy.yml` — 63 lines — mtime 2025-09-13 15:38:40 — sha256 `9daf10926641…`
- [19] `Dockerfile` — 27 lines — mtime 2025-09-13 16:37:38 — sha256 `5dc22d6e9ce9…`
- [20] `Makefile` — 42 lines — mtime 2025-09-13 16:38:26 — sha256 `2b0fac71f475…`
- [21] `pyproject.toml` — 22 lines — mtime 2025-09-12 23:43:45 — sha256 `d8c205569aa1…`
- [22] `render.yaml` — 16 lines — mtime 2025-09-13 17:51:23 — sha256 `ea92f0fb004c…`
- [23] `ruff.toml` — 19 lines — mtime 2025-09-13 18:38:03 — sha256 `13e62274b996…`
- [24] `tests/test_ask_providers.py` — 71 lines — mtime 2025-09-14 16:23:19 — sha256 `c46e4fb6e9cb…`
- [25] `tests/test_basic.py` — 22 lines — mtime 2025-09-13 18:33:31 — sha256 `552f7a87700f…`
- [26] `tests/test_duel_no_providers.py` — 18 lines — mtime 2025-09-14 11:42:30 — sha256 `dbbf7419d363…`
- [27] `tests/test_duel_openai_only.py` — 36 lines — mtime 2025-09-14 16:24:10 — sha256 `daabf14d47d9…`
- [28] `tests/test_fallback.py` — 57 lines — mtime 2025-09-14 16:31:51 — sha256 `8635780c76ce…`
- [29] `tests/test_metrics.py` — 21 lines — mtime 2025-09-13 18:33:31 — sha256 `856e3ec90817…`
- [30] `tests/test_metrics_error_counter.py` — 38 lines — mtime 2025-09-13 18:33:31 — sha256 `850ce9f5c170…`
- [31] `tests/test_observability.py` — 34 lines — mtime 2025-09-13 18:33:31 — sha256 `53d2b047fd81…`
- [32] `tests/test_openai_client.py` — 52 lines — mtime 2025-09-14 16:58:04 — sha256 `672c43b1e2b6…`
- [33] `tests/test_request_id.py` — 20 lines — mtime 2025-09-13 18:33:31 — sha256 `2ebc4c965dc3…`
- [34] `tests/test_request_id_header.py` — 16 lines — mtime 2025-09-14 11:44:44 — sha256 `f96698342a6f…`
- [35] `tools/guard_update.py` — 134 lines — mtime 2025-09-14 12:41:05 — sha256 `ad8e6b859128…`
- [36] `tools/snapshot_configs.py` — 290 lines — mtime 2025-09-14 12:04:40 — sha256 `6407d9d7f76f…`

---

## [1] .env.example
- Last modified: **2025-09-13 07:55:28**
- Lines: **29**
- SHA-256: `274aceefe36cbb2cf0840ec4f8ccc31f8ccc89a5909c6a2da3560dc705a88d2c`

```
# Variáveis futuras (preencher quando integrar IAs)

# .env.example
APP_NAME=orquestrador-ai
APP_VERSION=0.1.0
LOG_LEVEL=INFO

# Providers
OPENAI_API_KEY=coloque_sua_chave_aqui
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=coloque_sua_chave_aqui
GEMINI_MODEL=gemini-1.5-flash

# Orquestração
DEFAULT_PROVIDER=openai
PROVIDER_FALLBACK=openai,gemini

# Timeouts
HTTP_TIMEOUT=30
PROVIDER_TIMEOUT=25

# Cache (opcional)
REDIS_DSN=
CACHE_TTL_DEFAULT=60

# Métricas
METRICS_PATH=/metrics
```

## [2] .github/workflows/ci.yml
- Last modified: **2025-09-13 19:22:11**
- Lines: **39**
- SHA-256: `1741b400f7258048fbd5d71c4f9ddbd48860a00fe2da21a7f8b8232db8055ec5`

```yaml
name: CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [ "3.11", "3.13" ]
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install -e .
          pip install pytest ruff flake8

      - name: Lint (ruff)
        run: ruff check .

      - name: Lint (flake8)
        run: flake8 .

      - name: Tests
        run: pytest -q
```

## [3] .github/workflows/snapshot.yml
- Last modified: **2025-09-14 12:15:11**
- Lines: **21**
- SHA-256: `73f13efc74257069dcb4e3a8680af67d50367be462658b750527ff9dc7ddf703`

```yaml
name: Snapshot Guard
on:
  pull_request:
  push:
    branches: [main]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install --upgrade pip
      - run: python tools/snapshot_configs.py
      - name: Ensure snapshot is staged
        run: |
          git diff --exit-code CONFIG_SNAPSHOT.md CONFIG_SNAPSHOT.manifest.json || {
            echo "::error::Snapshot not up to date. Run tools/snapshot_configs.py and commit the result."
            exit 1
          }
```

## [4] .gitignore
- Last modified: **2025-09-12 01:56:56**
- Lines: **10**
- SHA-256: `b746016476e67e3218ae3f64849efa8088943b6fc34283a3dba598d296379793`

```
# Python
__pycache__/
*.pyc
.venv/
.env
# IDE
.vscode/
.idea/
# OSX
.DS_Store
```

## [5] app/__init__.py
- Last modified: **2025-09-13 15:30:17**
- Lines: **2**
- SHA-256: `f0fb5e1d3cbe63ad8149256a91c4b7228cbedfca932ffc0d9cb6086adee6c92f`

```python
# app/utils/__init__.py
# Torna 'utils' um pacote Python.
```

## [6] app/cache.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **34**
- SHA-256: `0a8edfe8a8567cffede09acee6554ec4f2dd543d756219a714b54ae786f2a475`

```python
from typing import Optional

import redis

from app.config import settings

_redis_client: Optional[redis.Redis] = None

def get_client() -> Optional[redis.Redis]:
    global _redis_client
    if _redis_client:
        return _redis_client
    if not settings.redis_url:
        return None
    _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client

def cache_get(key: str) -> Optional[str]:
    client = get_client()
    if not client:
        return None
    try:
        return client.get(key)
    except Exception:
        return None

def cache_set(key: str, value: str, ttl_seconds: int = 300) -> None:
    client = get_client()
    if not client:
        return
    try:
        client.setex(key, ttl_seconds, value)
    except Exception:
        pass
```

## [7] app/config.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **62**
- SHA-256: `ffa9c871ab06afa64e73084b2412d151470e59dd2c4089ac614ee6a2eec4b1d6`

```python
# app/config.py
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configurações centralizadas do orquestrador-ai.
    Carrega do ambiente e opcionalmente de um arquivo .env na raiz do projeto.
    """

    # App
    APP_NAME: str = Field(default="orquestrador-ai")
    APP_VERSION: str = Field(default="0.1.0")
    LOG_LEVEL: str = Field(default="INFO")  # DEBUG | INFO | WARNING | ERROR

    # Providers (Sprint 3)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")

    # Orquestração
    DEFAULT_PROVIDER: str = Field(default="openai")  # openai | gemini | echo
    PROVIDER_FALLBACK: List[str] = Field(
        default_factory=lambda: ["openai", "gemini"]
    )  # ordem de fallback

    # Timeouts (segundos)
    HTTP_TIMEOUT: float = Field(default=30.0)
    PROVIDER_TIMEOUT: float = Field(default=25.0)

    # Cache / Redis (opcional, para sprints futuras)
    REDIS_DSN: Optional[str] = Field(default=None)  # ex: redis://localhost:6379/0
    CACHE_TTL_DEFAULT: int = Field(default=60)  # segundos

    # Métricas
    METRICS_PATH: str = Field(default="/metrics")

    model_config = SettingsConfigDict(
        env_file=".env",           # carrega variáveis do .env se existir
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",            # ignora variáveis extras
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna uma instância única de Settings (cacheada).
    Use:
        from app.config import settings
    """
    return Settings()


# Instância pronta para import direto
settings = get_settings()
```

## [8] app/gemini_client.py
- Last modified: **2025-09-14 16:13:34**
- Lines: **83**
- SHA-256: `a75beba70841a25a6daeef509521a3aae94b746189590c890d0ae9db93b91843`

```python
# =============================================================================
# File: app/gemini_client.py
# Version: 2025-09-14 16:25:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO CRÍTICA: Corrigido SyntaxError ('{' was never closed) no dicionário de retorno.
# - Refatorado para ser totalmente assíncrono.
# - Utiliza `await gmodel.generate_content_async` para a chamada de API.
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, Optional

import google.generativeai as genai

from app.config import settings
from app.observability import logger


def is_configured() -> bool:
    """
    Retorna True se houver GEMINI_API_KEY configurada.
    """
    return bool(settings.GEMINI_API_KEY)


def _build_model(model_name: str):
    """
    Configura a SDK com API key e retorna a instância do modelo.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY não configurada.")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel(model_name)


async def ask_gemini(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,  # mantido para simetria
    **extra: Any,
) -> Dict[str, Any]:
    """
    Envia um prompt ao Gemini de forma assíncrona e retorna resposta normalizada.
    """
    mdl = model or settings.GEMINI_MODEL
    tmo = timeout or settings.PROVIDER_TIMEOUT

    logger.info("provider.gemini.request.async", model=mdl, timeout=tmo)

    try:
        gmodel = _build_model(mdl)

        generation_config = extra.get("generation_config")
        safety_settings = extra.get("safety_settings")

        resp = await gmodel.generate_content_async(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        content = (getattr(resp, "text", "") or "").strip()
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

        logger.info("provider.gemini.success.async", model=mdl)

        # DICIONÁRIO DE RETORNO CORRIGIDO
        return {
            "provider": "gemini",
            "model": mdl,
            "answer": content,
            "usage": usage,
        }
    except genai.types.generation_types.BlockedPromptException as e:
        logger.info("provider.gemini.blocked_prompt", error=str(e))
        raise RuntimeError("Prompt bloqueado pela política do Gemini.") from e
    except genai.types.generation_types.StopCandidateException as e:
        logger.info("provider.gemini.stop_candidate", error=str(e))
        raise RuntimeError("Geração interrompida pelo Gemini.") from e
    except Exception as e:
        logger.info("provider.gemini.unexpected_error", error=str(e))
        raise RuntimeError("Erro inesperado ao chamar o Gemini.") from e
```

## [9] app/judge.py
- Last modified: **2025-09-14 11:27:38**
- Lines: **71**
- SHA-256: `874ade3ba27169a304737fa60913fdb1e3068b4328a6d87c0ab1ee791c19794d`

```python
# app/judge.py
from __future__ import annotations
from typing import Dict, Any
from app.observability import logger
from app.openai_client import is_configured as openai_configured, ask_openai
from app.gemini_client import is_configured as gemini_configured, ask_gemini
import json

JUDGE_PROMPT = """Você é um avaliador técnico. Receba uma PERGUNTA e duas RESPOSTAS (A e B).
Analise CORREÇÃO, CLAREZA, SEGURANÇA e UTILIDADE.

Saída (JSON estrito, UMA linha):
{"winner":"A|B|tie","reason":"<explicação curta>"}

PERGUNTA:
{question}

RESPOSTA A:
{answer_a}

RESPOSTA B:
{answer_b}
"""

def _parse_simple_json_line(txt: str) -> Dict[str, Any]:
    line = (txt or "").strip()
    start = line.find("{")
    end = line.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(line[start:end+1])
            w = (obj.get("winner") or "tie").lower()
            if w not in ("a", "b", "tie"):
                w = "tie"
            return {"winner": w, "reason": obj.get("reason") or ""}
        except Exception:
            pass
    return {"winner": "tie", "reason": (line[:240] if line else "")}

def judge_answers(question: str, answer_a: str, answer_b: str) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b)

    if openai_configured():
        try:
            logger.info("judge.start", provider="openai")
            resp = ask_openai(prompt)
            parsed = _parse_simple_json_line(resp.get("answer"))
            logger.info("judge.done", provider="openai", winner=parsed["winner"])
            return {"provider": "openai", **parsed}
        except Exception as e:
            logger.info("judge.openai_error", error=str(e))

    if gemini_configured():
        try:
            logger.info("judge.start", provider="gemini")
            resp = ask_gemini(prompt)
            parsed = _parse_simple_json_line(resp.get("answer"))
            logger.info("judge.done", provider="gemini", winner=parsed["winner"])
            return {"provider": "gemini", **parsed}
        except Exception as e:
            logger.info("judge.gemini_error", error=str(e))

    # Heurística
    a_len = len((answer_a or "").strip())
    b_len = len((answer_b or "").strip())
    if a_len == 0 and b_len == 0:
        return {"provider": "heuristic", "winner": "tie", "reason": "Ambas vazias."}
    if abs(a_len - b_len) < 10:
        return {"provider": "heuristic", "winner": "tie", "reason": "Tamanho similar; empate técnico."}
    winner = "a" if a_len > b_len else "b"
    return {"provider": "heuristic", "winner": winner, "reason": "Resposta mais completa (comprimento)."}
```

## [10] app/main.py
- Last modified: **2025-09-14 16:18:50**
- Lines: **205**
- SHA-256: `560ecf5cacee15411d2fccfba1e10b974cf16fb4b4e122645474004da1031714`

```python
# =============================================================================
# File: app/main.py
# Version: 2025-09-14 16:45:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO CRÍTICA: Reescrevi o tratamento de erros para o mundo assíncrono.
# - Garante que HTTPExceptions sejam levantadas (raise) em vez de suprimidas.
# - Corrige o comportamento que retornava 200 OK em caso de falha.
# - Assegura que o modo duelo lida corretamente com falha de um dos provedores.
# =============================================================================

from __future__ import annotations
import time
import asyncio
from typing import Any, Dict, List, Tuple, Optional

from fastapi import FastAPI, Body, HTTPException

from app.config import settings
from app.metrics import setup_metrics, record_ask
from app.observability import RequestIDMiddleware, TraceMiddleware, logger
from app.openai_client import ask_openai, is_configured as openai_configured
from app.gemini_client import ask_gemini, is_configured as gemini_configured
from app.judge import judge_answers

app = FastAPI(
    title="orquestrador-ai",
    version=settings.APP_VERSION,
    description="Orquestrador multi-IA com observabilidade e métricas",
)

# --- Middlewares e métricas ---
setup_metrics(app)
app.add_middleware(TraceMiddleware)
app.add_middleware(RequestIDMiddleware)

# --- Rotas de infraestrutura ---
@app.get("/", tags=["infra"])
async def root():
    logger.info("root.live")
    return {"status": "live"}

@app.get("/health", tags=["infra"])
async def health():
    logger.info("health.ok")
    return {"status": "ok"}

@app.get("/ready", tags=["infra"])
async def readiness():
    logger.info("readiness.ok")
    return {"status": "ready"}


# --- Lógica dos Provedores ---
def _provider_is_configured(name: str) -> bool:
    n = (name or "").lower()
    if n == "openai":
        return openai_configured()
    if n == "gemini":
        return gemini_configured()
    if n == "echo":
        return True
    return False


async def _provider_call(name: str, prompt: str) -> Dict[str, Any]:
    n = (name or "").lower()

    if n == "echo":
        logger.info("ask.echo", prompt=prompt)
        return {"provider": "echo", "answer": prompt, "output": prompt}
    
    if not _provider_is_configured(n):
        detail = f"{n.upper()}_API_KEY não configurada." if n in ("openai", "gemini") else f"Provider não suportado: {name}"
        raise HTTPException(status_code=503, detail=detail)

    try:
        if n == "openai":
            return await ask_openai(prompt)
        if n == "gemini":
            return await ask_gemini(prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    raise HTTPException(status_code=400, detail=f"Provider não suportado: {name}")


def _fallback_chain() -> List[str]:
    return ["openai", "gemini"]


# --- Lógica do Modo Duelo ---
async def _try_call(p: str, prompt: str) -> Tuple[str, Dict[str, Any] | None, str | None]:
    try:
        if not _provider_is_configured(p):
            return p, None, "não configurado"
        return p, await _provider_call(p, prompt), None
    except HTTPException as http_exc:
        return p, None, f"http_{http_exc.status_code}: {http_exc.detail}"
    except Exception as e:
        return p, None, f"erro: {e}"


def _duel_error(reason: str, results: Dict, errors: Dict) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail={
            "mode": "duel",
            "reason": reason,
            "responses": {
                "openai": {"ok": results.get("openai") is not None, "answer": (results.get("openai") or {}).get("answer"), "error": errors.get("openai")},
                "gemini": {"ok": results.get("gemini") is not None, "answer": (results.get("gemini") or {}).get("answer"), "error": errors.get("gemini")},
            },
            "verdict": {"winner": "none"},
        },
    )


async def _ask_duel(prompt: str) -> Dict[str, Any]:
    tasks = [_try_call("openai", prompt), _try_call("gemini", prompt)]
    results_tuples = await asyncio.gather(*tasks, return_exceptions=True)

    results: Dict[str, Dict[str, Any] | None] = {}
    errors: Dict[str, str | None] = {}
    
    for result in results_tuples:
        if isinstance(result, Exception):
            logger.error("duel.gather.exception", error=str(result))
            continue
        prov, resp, err = result
        results[prov] = resp
        errors[prov] = err

    if not openai_configured() and not gemini_configured():
         raise _duel_error("nenhum provider configurado", results, errors)

    a = (results.get("openai") or {}).get("answer") or ""
    b = (results.get("gemini") or {}).get("answer") or ""

    if not a and not b:
        raise _duel_error("nenhum provider retornou conteúdo", results, errors)

    verdict_llm = await judge_answers(prompt, a, b)
    winner_map = {"a": "openai", "b": "gemini", "tie": "tie"}
    raw_winner = (verdict_llm or {}).get("winner")
    winner = winner_map.get(raw_winner, "tie")
    verdict: Dict[str, Any] = {"winner": winner, "rationale": (verdict_llm or {}).get("reason")}

    return {
        "mode": "duel", "prompt": prompt,
        "responses": {
            "openai": {"ok": results.get("openai") is not None, "answer": a if a else None, "error": errors.get("openai")},
            "gemini": {"ok": results.get("gemini") is not None, "answer": b if b else None, "error": errors.get("gemini")},
        },
        "verdict": verdict,
    }


# --- Rotas de Negócio Principais ---
@app.post("/ask", tags=["ask"])
async def ask(provider: str = "auto", payload: dict = Body(...), use_fallback: bool = True):
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório no corpo JSON.")

    effective_provider = (provider or "auto").lower()

    if effective_provider == "duel":
        return await _ask_duel(prompt)

    start_time = time.perf_counter()
    
    if effective_provider != "auto":
        try:
            resp = await _provider_call(effective_provider, prompt)
            record_ask(effective_provider, "success", (time.perf_counter() - start_time) * 1000)
            return resp
        except HTTPException as e:
            record_ask(effective_provider, "error", (time.perf_counter() - start_time) * 1000)
            raise e

    chain = _fallback_chain()
    last_error: Optional[HTTPException] = None
    for p in chain:
        try:
            resp = await _provider_call(p, prompt)
            record_ask(p, "success", (time.perf_counter() - start_time) * 1000)
            return resp
        except HTTPException as e:
            record_ask(p, "error", (time.perf_counter() - start_time) * 1000)
            last_error = e
            if not use_fallback:
                break
    
    if last_error:
        raise last_error
    raise HTTPException(status_code=502, detail="Falha ao atender requisição em todos os provedores.")


@app.post("/duel", tags=["duel"])
async def duel(payload: dict = Body(...)):
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório no corpo JSON.")
    return await _ask_duel(prompt)
```

## [11] app/metrics.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **44**
- SHA-256: `672d96010d9c1976479acbd424570f61827b8e97636e8e1089dfed729ac9ba8c`

```python
# app/metrics.py
from typing import Optional

from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# Contador de requisições do /ask por provider e status (success|error)
ASK_REQUESTS_TOTAL = Counter(
    "ask_requests_total",
    "Total de chamadas ao /ask",
    labelnames=("provider", "status"),
)

# Latência do /ask por provider e status (em segundos)
ASK_LATENCY_SECONDS = Histogram(
    "ask_latency_seconds",
    "Latência das chamadas ao /ask (s)",
    labelnames=("provider", "status"),
)


def setup_metrics(app, endpoint: str = "/metrics"):
    """
    Instrumenta a app FastAPI e expõe /metrics.
    """
    Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint=endpoint)


def record_ask(provider: str, status: str, duration_ms: Optional[float] = None) -> None:
    """
    Registra uma ocorrência do /ask nas métricas personalizadas.
    - provider: "echo" | "openai" | "gemini" | ...
    - status: "success" | "error" (ou outro rótulo que desejar padronizar)
    - duration_ms: opcional; se fornecido, registra no histograma em segundos
    """
    p = (provider or "unknown").lower()
    s = (status or "unknown").lower()

    # incrementa contador
    ASK_REQUESTS_TOTAL.labels(provider=p, status=s).inc()

    # observa latência se fornecida
    if duration_ms is not None:
        ASK_LATENCY_SECONDS.labels(provider=p, status=s).observe(duration_ms / 1000.0)
```

## [12] app/observability.py
- Last modified: **2025-09-14 11:32:26**
- Lines: **111**
- SHA-256: `5a5528e1c8ab181c8bdc64133f5ebf08f502ee930d15dc116eabe9ed7e8fb297`

```python
# app/observability.py
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from typing import Callable, Optional

import structlog
from structlog.contextvars import bind_contextvars, merge_contextvars, clear_contextvars
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# -------- Config de log / structlog --------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REQUEST_ID_HEADER = "X-Request-ID"

def _configure_logger():
    logging.basicConfig(
        stream=sys.stdout,
        format="%(message)s",
        level=getattr(logging, LOG_LEVEL, logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.stdlib.add_log_level,
            # injeta os contextvars (inclui request_id quando houver)
            merge_contextvars,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("orquestrador-ai")

logger = _configure_logger()

# -------- Middlewares --------
class TraceMiddleware(BaseHTTPMiddleware):
    """
    Observabilidade de requisições:
    - Loga início e fim
    - Calcula duração (ms)
    *Não* garante X-Request-ID (isso é do RequestIDMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response: Optional[Response] = None
        try:
            logger.info("request.start", path=str(request.url), method=request.method)
            response = await call_next(request)
            return response
        finally:
            dur_ms = (time.perf_counter() - start) * 1000
            status = getattr(response, "status_code", None) if response is not None else None
            logger.info(
                "request.end",
                path=str(request.url),
                method=request.method,
                status=status,
                duration_ms=round(dur_ms, 2),
            )

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Garante e propaga o X-Request-ID:
    - Lê do request (aceita 'X-Request-ID' ou 'x-request-id')
    - Gera UUID4 se ausente
    - Sempre escreve 'X-Request-ID' na resposta
    - Expõe em request.state.request_id
    - Faz bind no structlog contextvars pro ID aparecer nos logs
    """

    def __init__(self, app, header_name: str = REQUEST_ID_HEADER):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get(self.header_name) or request.headers.get(self.header_name.lower())
        request_id = incoming or str(uuid.uuid4())

        # Disponibiliza no state e no contexto do logger
        setattr(request.state, "request_id", request_id)
        bind_contextvars(request_id=request_id)

        response: Optional[Response] = None
        try:
            response = await call_next(request)
            return response
        finally:
            # Garante o header SEMPRE, mesmo em exceção
            if response is None:
                response = Response()
            response.headers[self.header_name] = request_id
            # Evita vazar contexto para a próxima request
            clear_contextvars()
            # Como estamos no finally, precisamos devolver a response
            # (o Starlette espera que retornemos a mesma instância criada aqui)
            # Portanto, só retornamos se não retornamos antes
            if not hasattr(response, "_already_returned"):  # flag defensiva
                response._already_returned = True  # type: ignore[attr-defined]
                return response

__all__ = ["logger", "TraceMiddleware", "RequestIDMiddleware"]
```

## [13] app/openai_client.py
- Last modified: **2025-09-14 16:00:55**
- Lines: **114**
- SHA-256: `8526780b124e3f7a168702fe71762412b0513bce96727071e8598ff16a6d4be7`

```python
# =============================================================================
# File: app/openai_client.py
# Version: 2025-09-14 15:59:00 -03 (America/Sao_Paulo)
# Changes:
# - Refatorado para ser totalmente assíncrono.
# - Uso do AsyncOpenAI para chamadas não-bloqueantes.
# - Função _build_async_client para criar o cliente assíncrono.
# - ask_openai agora é uma função `async def`.
# - Utiliza `await client.chat.completions.create` para a chamada de API.
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, Optional

from openai import APIConnectionError, APIStatusError, AuthenticationError, AsyncOpenAI, RateLimitError

from app.config import settings
from app.observability import logger


def is_configured() -> bool:
    """
    Retorna True se houver OPENAI_API_KEY configurada.
    """
    return bool(settings.OPENAI_API_KEY)


def _build_async_client(timeout: Optional[float] = None) -> AsyncOpenAI:
    """
    Constroi o cliente AsyncOpenAI com a API key do settings.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não configurada.")

    # Usa o cliente assíncrono para chamadas não-bloqueantes
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=timeout or settings.PROVIDER_TIMEOUT)


async def ask_openai(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Envia um prompt para o OpenAI Chat Completions de forma assíncrona.

    Parâmetros:
      - prompt: texto do usuário
      - model: override do modelo (default: settings.OPENAI_MODEL)
      - timeout: timeout em segundos para esta chamada (default: settings.PROVIDER_TIMEOUT)
      - **extra: espaço para parâmetros futuros (temperature, top_p, etc.)

    Retorno:
      {
        "provider": "openai",
        "model": "<modelo>",
        "answer": "<texto>",
        "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
      }

    Exceções:
      - RuntimeError em casos de erro de conexão, auth, rate limit, status != 2xx, ou outro erro inesperado.
    """
    mdl = model or settings.OPENAI_MODEL
    tmo = timeout or settings.PROVIDER_TIMEOUT

    client = _build_async_client(timeout=tmo)
    logger.info("provider.openai.request.async", model=mdl)

    try:
        # A biblioteca openai gerencia retries para o cliente async por padrão
        resp = await client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )

        content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

        logger.info(
            "provider.openai.success.async",
            model=mdl,
            total_tokens=usage["total_tokens"],
        )

        return {
            "provider": "openai",
            "model": mdl,
            "answer": content,
            "usage": usage,
        }

    except AuthenticationError as e:
        logger.info("provider.openai.auth_error", error=str(e))
        raise RuntimeError("Falha de autenticação na OpenAI (verifique OPENAI_API_KEY).") from e
    except RateLimitError as e:
        logger.info("provider.openai.rate_limit", error=str(e))
        raise RuntimeError("Rate limit atingido na OpenAI. Tente novamente mais tarde.") from e
    except APIStatusError as e:
        logger.info("provider.openai.api_status_error", status=e.status_code, error=str(e))
        raise RuntimeError(f"Erro de status na OpenAI: {e.status_code}.") from e
    except APIConnectionError as e:
        logger.info("provider.openai.connection_error", error=str(e))
        raise RuntimeError("Erro de conexão com a OpenAI.") from e
    except Exception as e:
        logger.info("provider.openai.unexpected_error", error=str(e))
        raise RuntimeError("Erro inesperado ao chamar a OpenAI.") from e
```

## [14] app/retry.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **42**
- SHA-256: `f7539c2a26731371f8fb49dccb0d0a05cd996742ba73131cdf641e7334591a5b`

```python
# app/utils/retry.py
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Type


class RetryExceededError(RuntimeError):
    """Lançado quando todas as tentativas de retry se esgotam."""
    pass


def retry(
    fn: Callable[[], Any],
    retries: int = 2,
    backoff_ms: int = 200,
    retry_on: Tuple[Type[BaseException], ...] = (TimeoutError, ConnectionError),
    sleep: Optional[Callable[[float], None]] = None,
) -> Any:
    """
    Executa `fn` com tentativas de retry em erros transitórios.

    - retries: número de novas tentativas após a primeira (total de chamadas = 1 + retries)
    - backoff_ms: atraso (milissegundos) entre tentativas, exponencial (x2) a cada falha
    - retry_on: tupla de exceções consideradas transitórias para retry
    - sleep: função de espera (recebe segundos). Se None, não dorme (útil para testes).

    Retorna o valor de `fn` na primeira execução bem-sucedida ou lança o último erro.
    """
    attempts = 0
    delay_sec = max(backoff_ms, 0) / 1000.0

    while True:
        try:
            attempts += 1
            return fn()
        except retry_on as exc:
            if attempts > retries:
                raise RetryExceededError(f"Tentativas esgotadas ({attempts})") from exc
            if sleep:
                sleep(delay_sec)
            # backoff exponencial simples
            delay_sec = delay_sec * 2 if delay_sec > 0 else 0.0
```

## [15] app/utils/__init__.py
- Last modified: **2025-09-13 15:31:19**
- Lines: **2**
- SHA-256: `f0fb5e1d3cbe63ad8149256a91c4b7228cbedfca932ffc0d9cb6086adee6c92f`

```python
# app/utils/__init__.py
# Torna 'utils' um pacote Python.
```

## [16] app/utils/retry.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **42**
- SHA-256: `d22081dab42b13ce05a5ee87d5aa66ac14ed40dc07c55221d46d39b353cdfd9c`

```python
# app/utils/retry.py
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Type


class RetryExceededError(RuntimeError):
    """Lançado quando todas as tentativas de retry se esgotam."""
    pass


def retry(
    fn: Callable[[], Any],
    retries: int = 2,
    backoff_ms: int = 200,
    retry_on: Tuple[Type[BaseException], ...] = (TimeoutError, ConnectionError),
    sleep: Optional[Callable[[float], None]] = None,
) -> Any:
    """
    Executa `fn` com tentativas de retry em erros transitórios.

    - retries: novas tentativas após a primeira (total = 1 + retries)
    - backoff_ms: atraso (ms) entre tentativas; exponencial (x2)
    - retry_on: exceções que disparam retry
    - sleep: função que recebe segundos (permite no-op em testes)

    Retorna o valor de `fn` na primeira execução bem-sucedida
    ou lança o último erro após esgotar as tentativas.
    """
    attempts = 0
    delay_sec = max(backoff_ms, 0) / 1000.0

    while True:
        try:
            attempts += 1
            return fn()
        except retry_on as exc:
            if attempts > retries:
                raise RetryExceededError(f"Tentativas esgotadas ({attempts})") from exc
            if sleep:
                sleep(delay_sec)
            delay_sec = delay_sec * 2 if delay_sec > 0 else 0.0
```

## [17] CONFIG_SNAPSHOT.manifest.json
- Last modified: **2025-09-14 17:21:00**
- Lines: **225**
- SHA-256: `48ae3dd7a70cc9a14303a6a7c8bdc93c5de4ded6c98eed6b0433cc5576559fdc`

```json
{
  "generated_at": "2025-09-14 17:21:00 ",
  "root": "/Users/wagnerjfjunior/orquestrador-ai",
  "file_count": 36,
  "total_lines": 2134,
  "hash_algorithm": "sha256",
  "files": [
    {
      "path": ".env.example",
      "mtime": "2025-09-13 07:55:28",
      "lines": 29,
      "sha256": "274aceefe36cbb2cf0840ec4f8ccc31f8ccc89a5909c6a2da3560dc705a88d2c"
    },
    {
      "path": ".github/workflows/ci.yml",
      "mtime": "2025-09-13 19:22:11",
      "lines": 39,
      "sha256": "1741b400f7258048fbd5d71c4f9ddbd48860a00fe2da21a7f8b8232db8055ec5"
    },
    {
      "path": ".github/workflows/snapshot.yml",
      "mtime": "2025-09-14 12:15:11",
      "lines": 21,
      "sha256": "73f13efc74257069dcb4e3a8680af67d50367be462658b750527ff9dc7ddf703"
    },
    {
      "path": ".gitignore",
      "mtime": "2025-09-12 01:56:56",
      "lines": 10,
      "sha256": "b746016476e67e3218ae3f64849efa8088943b6fc34283a3dba598d296379793"
    },
    {
      "path": "app/__init__.py",
      "mtime": "2025-09-13 15:30:17",
      "lines": 2,
      "sha256": "f0fb5e1d3cbe63ad8149256a91c4b7228cbedfca932ffc0d9cb6086adee6c92f"
    },
    {
      "path": "app/cache.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 34,
      "sha256": "0a8edfe8a8567cffede09acee6554ec4f2dd543d756219a714b54ae786f2a475"
    },
    {
      "path": "app/config.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 62,
      "sha256": "ffa9c871ab06afa64e73084b2412d151470e59dd2c4089ac614ee6a2eec4b1d6"
    },
    {
      "path": "app/gemini_client.py",
      "mtime": "2025-09-14 16:13:34",
      "lines": 83,
      "sha256": "a75beba70841a25a6daeef509521a3aae94b746189590c890d0ae9db93b91843"
    },
    {
      "path": "app/judge.py",
      "mtime": "2025-09-14 11:27:38",
      "lines": 71,
      "sha256": "874ade3ba27169a304737fa60913fdb1e3068b4328a6d87c0ab1ee791c19794d"
    },
    {
      "path": "app/main.py",
      "mtime": "2025-09-14 16:18:50",
      "lines": 205,
      "sha256": "560ecf5cacee15411d2fccfba1e10b974cf16fb4b4e122645474004da1031714"
    },
    {
      "path": "app/metrics.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 44,
      "sha256": "672d96010d9c1976479acbd424570f61827b8e97636e8e1089dfed729ac9ba8c"
    },
    {
      "path": "app/observability.py",
      "mtime": "2025-09-14 11:32:26",
      "lines": 111,
      "sha256": "5a5528e1c8ab181c8bdc64133f5ebf08f502ee930d15dc116eabe9ed7e8fb297"
    },
    {
      "path": "app/openai_client.py",
      "mtime": "2025-09-14 16:00:55",
      "lines": 114,
      "sha256": "8526780b124e3f7a168702fe71762412b0513bce96727071e8598ff16a6d4be7"
    },
    {
      "path": "app/retry.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 42,
      "sha256": "f7539c2a26731371f8fb49dccb0d0a05cd996742ba73131cdf641e7334591a5b"
    },
    {
      "path": "app/utils/__init__.py",
      "mtime": "2025-09-13 15:31:19",
      "lines": 2,
      "sha256": "f0fb5e1d3cbe63ad8149256a91c4b7228cbedfca932ffc0d9cb6086adee6c92f"
    },
    {
      "path": "app/utils/retry.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 42,
      "sha256": "d22081dab42b13ce05a5ee87d5aa66ac14ed40dc07c55221d46d39b353cdfd9c"
    },
    {
      "path": "CONFIG_SNAPSHOT.manifest.json",
      "mtime": "2025-09-14 17:20:47",
      "lines": 225,
      "sha256": "4301b1facb788f1886085f482d63d3182614330928fb488da560f398228acdbc"
    },
    {
      "path": "cy.yml",
      "mtime": "2025-09-13 15:38:40",
      "lines": 63,
      "sha256": "9daf109266413c593e79e83e307681f1bc2533105d6fe072cb680e42068115ee"
    },
    {
      "path": "Dockerfile",
      "mtime": "2025-09-13 16:37:38",
      "lines": 27,
      "sha256": "5dc22d6e9ce98b3471ee6c36b3a68fc9776bb34aa909aea118fe32659ec182d3"
    },
    {
      "path": "Makefile",
      "mtime": "2025-09-13 16:38:26",
      "lines": 42,
      "sha256": "2b0fac71f475fec513c9d68944302b988395b27ce6d32189cd85a4072287cdac"
    },
    {
      "path": "pyproject.toml",
      "mtime": "2025-09-12 23:43:45",
      "lines": 22,
      "sha256": "d8c205569aa1662debd668a45c32f50780aff2e6daf30fe611f3ca155461bf93"
    },
    {
      "path": "render.yaml",
      "mtime": "2025-09-13 17:51:23",
      "lines": 16,
      "sha256": "ea92f0fb004c850d21ff2c2e5cca496accdcbda2cc8b7eb11ab9c4f483748420"
    },
    {
      "path": "ruff.toml",
      "mtime": "2025-09-13 18:38:03",
      "lines": 19,
      "sha256": "13e62274b99610f74f7908bf2ad18652901bc828ff3425552f32f31504c6ed0c"
    },
    {
      "path": "tests/test_ask_providers.py",
      "mtime": "2025-09-14 16:23:19",
      "lines": 71,
      "sha256": "c46e4fb6e9cb402bd467eee5bb009603e0d1eb65394e421301770e25dd08981a"
    },
    {
      "path": "tests/test_basic.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 22,
      "sha256": "552f7a87700f92408da2d70adde4e8a9f7ac467594a502ede50e3e0ebe75ec9a"
    },
    {
      "path": "tests/test_duel_no_providers.py",
      "mtime": "2025-09-14 11:42:30",
      "lines": 18,
      "sha256": "dbbf7419d36344394b3bc7c5a9f406c0847f7b236843b1f01e2b155dad732052"
    },
    {
      "path": "tests/test_duel_openai_only.py",
      "mtime": "2025-09-14 16:24:10",
      "lines": 36,
      "sha256": "daabf14d47d900829188e01b110cd52087b2b607566e63d9bc63c5b0e08dea79"
    },
    {
      "path": "tests/test_fallback.py",
      "mtime": "2025-09-14 16:31:51",
      "lines": 57,
      "sha256": "8635780c76ceab5bf3dee8cc6925fefe03c4138a4701c9be0f7f61c1ccba8f4d"
    },
    {
      "path": "tests/test_metrics.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 21,
      "sha256": "856e3ec90817539595c20dfa86ad6eaf05a449d356fc23214aa655621d470784"
    },
    {
      "path": "tests/test_metrics_error_counter.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 38,
      "sha256": "850ce9f5c170c26e823a27370a8b2374b6f755dc6ba6af0740ac3da54b92c58a"
    },
    {
      "path": "tests/test_observability.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 34,
      "sha256": "53d2b047fd810fb276cdb3d760cc915c7f70e908c6a5dc9dbe730d7a21d0b145"
    },
    {
      "path": "tests/test_openai_client.py",
      "mtime": "2025-09-14 16:58:04",
      "lines": 52,
      "sha256": "672c43b1e2b6b573e190b75f88338f9d03a7ea2175e51229a2cdef1b8ab5bdc6"
    },
    {
      "path": "tests/test_request_id.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 20,
      "sha256": "2ebc4c965dc3223c112ce5d370ddd116ed9389db792237d8c1574becffef2452"
    },
    {
      "path": "tests/test_request_id_header.py",
      "mtime": "2025-09-14 11:44:44",
      "lines": 16,
      "sha256": "f96698342a6fa9152826a1d9e66d5acba5590c19ed5735b0137ddc68bc89db2b"
    },
    {
      "path": "tools/guard_update.py",
      "mtime": "2025-09-14 12:41:05",
      "lines": 134,
      "sha256": "ad8e6b85912848227fe0d097e22e08ed8813334959883fd6709ff969f350b738"
    },
    {
      "path": "tools/snapshot_configs.py",
      "mtime": "2025-09-14 12:04:40",
      "lines": 290,
      "sha256": "6407d9d7f76f4bdf2e7626bb17061d46e5b91620495b879e277f48dba4aba0c2"
    }
  ]
}
```

## [18] cy.yml
- Last modified: **2025-09-13 15:38:40**
- Lines: **63**
- SHA-256: `9daf109266413c593e79e83e307681f1bc2533105d6fe072cb680e42068115ee`

```yaml
name: CI

on:
  push:
    branches: [ "main", "develop", "feature/**", "fix/**" ]
  pull_request:
    branches: [ "main", "develop" ]

jobs:
  test:
    name: Lint & Tests (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.13"]

    env:
      # Garante que não usamos chaves reais no CI
      OPENAI_API_KEY: ""
      GEMINI_API_KEY: ""
      # Faz o Python preferir stdout sem buffer — logs mais legíveis
      PYTHONUNBUFFERED: "1"
      # Garante que possamos importar o pacote local
      PYTHONPATH: "."

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Show Python version
        run: python -V

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          # Se houver requirements.txt, usa; senão instala mínimo e modo dev local
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          fi
          # Dependências mínimas para o projeto e CI
          pip install -e .
          pip install pytest ruff flake8

      - name: Ruff (lint)
        run: |
          # Se houver config (ruff.toml/pyproject), ela será respeitada
          ruff check .

      - name: Flake8 (style)
        run: |
          # Ajuste output mínimo; personalize ignore/max-line-length em setup.cfg/pyproject se quiser
          flake8 .

      - name: Run tests
        run: |
          pytest -q
```

## [19] Dockerfile
- Last modified: **2025-09-13 16:37:38**
- Lines: **27**
- SHA-256: `5dc22d6e9ce98b3471ee6c36b3a68fc9776bb34aa909aea118fe32659ec182d3`

```dockerfile
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
```

## [20] Makefile
- Last modified: **2025-09-13 16:38:26**
- Lines: **42**
- SHA-256: `2b0fac71f475fec513c9d68944302b988395b27ce6d32189cd85a4072287cdac`

```makefile
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
```

## [21] pyproject.toml
- Last modified: **2025-09-12 23:43:45**
- Lines: **22**
- SHA-256: `d8c205569aa1662debd668a45c32f50780aff2e6daf30fe611f3ca155461bf93`

```toml
[project]
name = "orquestrador-ai"
version = "0.1.0"
description = "Orquestrador adaptativo (esqueleto)"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.112",
  "uvicorn[standard]>=0.30",
  "pydantic-settings>=2.4.0",
  "redis>=5.0",
  "openai>=1.40.0",
  "google-generativeai>=0.7.0",
  "httpx>=0.27.0",
  "structlog>=24.1.0",
  "prometheus-fastapi-instrumentator>=7.0.0",
]

[tool.uvicorn]
factory = false
host = "0.0.0.0"
port = 8080
reload = true
```

## [22] render.yaml
- Last modified: **2025-09-13 17:51:23**
- Lines: **16**
- SHA-256: `ea92f0fb004c850d21ff2c2e5cca496accdcbda2cc8b7eb11ab9c4f483748420`

```yaml
# render.yaml
services:
  - type: web
    name: orquestrador-ai
    env: docker
    plan: free
    region: oregon
    autoDeploy: true
    healthCheckPath: /health
    envVars:
      - key: LOG_LEVEL
        value: INFO
      - key: OPENAI_API_KEY
        sync: false   # configure no dashboard/secret
      - key: GEMINI_API_KEY
        sync: false
```

## [23] ruff.toml
- Last modified: **2025-09-13 18:38:03**
- Lines: **19**
- SHA-256: `13e62274b99610f74f7908bf2ad18652901bc828ff3425552f32f31504c6ed0c`

```toml
# ruff.toml

# topo (quando arquivo é ruff.toml, sem [tool.ruff])
line-length = 120
target-version = "py311"

[lint]
# E,F: pycodestyle/pyflakes | I: isort (organiza imports)
select = ["E", "F", "I"]
ignore = []

# Ignorar nos testes:
# - F401 (imports não usados, ex.: pytest)
# - E501 (linhas longas em strings/asserts)
per-file-ignores = { "tests/**" = ["F401", "E501"] }

[lint.isort]
# trata "app" como first-party ao organizar imports
known-first-party = ["app"]
```

## [24] tests/test_ask_providers.py
- Last modified: **2025-09-14 16:23:19**
- Lines: **71**
- SHA-256: `c46e4fb6e9cb402bd467eee5bb009603e0d1eb65394e421301770e25dd08981a`

```python
# =============================================================================
# File: tests/test_ask_providers.py
# Version: 2025-09-14 16:45:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO: As funções de mock `fake_ask_...` foram convertidas para `async def`.
# - Isso é necessário para que o monkeypatch funcione com o novo código assíncrono.
# =============================================================================
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ask_openai_success(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: True)

    async def fake_ask_openai(prompt):  # <-- MUDANÇA: async def
        assert prompt == "olá openai"
        return {
            "provider": "openai", "model": "gpt-4o-mini", "answer": "oi, daqui é o openai",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("app.main.ask_openai", fake_ask_openai)

    r = client.post("/ask?provider=openai", json={"prompt": "olá openai"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "openai"
    assert data["answer"] == "oi, daqui é o openai"


def test_ask_gemini_success(monkeypatch):
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    async def fake_ask_gemini(prompt):  # <-- MUDANÇA: async def
        assert prompt == "olá gemini"
        return {
            "provider": "gemini", "model": "gemini-1.5-flash", "answer": "oi, daqui é o gemini",
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }

    monkeypatch.setattr("app.main.ask_gemini", fake_ask_gemini)

    r = client.post("/ask?provider=gemini", json={"prompt": "olá gemini"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "gemini"
    assert data["answer"] == "oi, daqui é o gemini"


def test_ask_openai_not_configured(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: False)
    r = client.post("/ask?provider=openai", json={"prompt": "qualquer"})
    assert r.status_code == 503
    assert "não configurada" in r.json()["detail"].lower()


def test_ask_gemini_provider_error(monkeypatch):
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    async def boom(prompt):  # <-- MUDANÇA: async def
        raise RuntimeError("Rate limit atingido")

    monkeypatch.setattr("app.main.ask_gemini", boom)

    r = client.post("/ask?provider=gemini", json={"prompt": "teste"})
    assert r.status_code == 502
    assert "limit" in r.json()["detail"].lower()
```

## [25] tests/test_basic.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **22**
- SHA-256: `552f7a87700f92408da2d70adde4e8a9f7ac467594a502ede50e3e0ebe75ec9a`

```python
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "live"

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ask_echo():
    r = client.post("/ask?provider=echo", json={"prompt": "ping"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "echo"
    assert data["answer"] == "ping"
```

## [26] tests/test_duel_no_providers.py
- Last modified: **2025-09-14 11:42:30**
- Lines: **18**
- SHA-256: `dbbf7419d36344394b3bc7c5a9f406c0847f7b236843b1f01e2b155dad732052`

```python
# tests/test_duel_no_providers.py
from fastapi.testclient import TestClient
import app.main as m

client = TestClient(m.app)

def test_duel_returns_502_when_no_providers():
    # Nenhum provider “configurado”
    m.openai_configured = lambda: False
    m.gemini_configured = lambda: False

    resp = client.post("/duel", json={"prompt": "qual a capital da França?"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["detail"]["mode"] == "duel"
    assert body["detail"]["verdict"]["winner"] == "none"
    assert "openai" in body["detail"]["responses"]
    assert "gemini" in body["detail"]["responses"]
```

## [27] tests/test_duel_openai_only.py
- Last modified: **2025-09-14 16:24:10**
- Lines: **36**
- SHA-256: `daabf14d47d900829188e01b110cd52087b2b607566e63d9bc63c5b0e08dea79`

```python
# =============================================================================
# File: tests/test_duel_openai_only.py
# Version: 2025-09-14 16:45:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO: A função de mock `_fake_provider_call` foi convertida para `async def`.
# - CORREÇÃO: O mock de `judge_answers` também precisa ser `async`.
# - O teste agora espera 200 OK, pois o duelo deve funcionar com apenas 1 provedor.
# =============================================================================
from fastapi.testclient import TestClient
import app.main as m

client = TestClient(m.app)

def test_duel_openai_only_ok(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: True)
    monkeypatch.setattr("app.main.gemini_configured", lambda: False)

    async def _fake_provider_call(name, prompt): # <-- MUDANÇA: async def
        if name == "openai":
            return {"provider": "openai", "answer": "Paris é a capital da França."}
        raise RuntimeError("Gemini not configured")
    
    async def fake_judge(q, a, b): # <-- MUDANÇA: async def
        return {"winner": "a", "reason": "A is valid"}

    monkeypatch.setattr(m, "_provider_call", _fake_provider_call)
    monkeypatch.setattr(m, "judge_answers", fake_judge)

    resp = client.post("/duel", json={"prompt": "qual a capital da França?"})
    
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "duel"
    assert body["responses"]["openai"]["ok"] is True
    assert "Paris" in (body["responses"]["openai"]["answer"] or "")
    assert body["verdict"]["winner"] in ("openai", "tie")
```

## [28] tests/test_fallback.py
- Last modified: **2025-09-14 16:31:51**
- Lines: **57**
- SHA-256: `8635780c76ceab5bf3dee8cc6925fefe03c4138a4701c9be0f7f61c1ccba8f4d`

```python
# =============================================================================
# File: tests/test_fallback.py
# Version: 2025-09-14 16:30:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO: As funções de mock `boom` e `ok` foram convertidas para `async def`.
# - CORREÇÃO: `test_fallback_todos_falham` agora espera a mensagem de erro correta.
# - As funções de mock dos provedores agora precisam ser assíncronas.
# =============================================================================
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_fallback_openai_falha_e_gemini_sucesso(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: True)
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    async def boom(prompt):  # <-- MUDANÇA: async def
        raise RuntimeError("Erro simulado no OpenAI")

    async def ok(prompt):  # <-- MUDANÇA: async def
        return {"provider": "gemini", "model": "gemini-1.5-flash", "answer": "ok gemini", "usage": {}}

    monkeypatch.setattr("app.main.ask_openai", boom)
    monkeypatch.setattr("app.main.ask_gemini", ok)

    r = client.post("/ask?provider=auto", json={"prompt": "hi"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "gemini"
    assert data["answer"] == "ok gemini"


def test_fallback_provider_explicito_sem_fallback(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: True)

    async def boom(prompt):  # <-- MUDANÇA: async def
        raise RuntimeError("Erro simulado no OpenAI")

    monkeypatch.setattr("app.main.ask_openai", boom)

    r = client.post("/ask?provider=openai&use_fallback=false", json={"prompt": "hi"})
    assert r.status_code == 502
    assert "erro" in r.json()["detail"].lower()


def test_fallback_todos_falham(monkeypatch):
    monkeypatch.setattr("app.main.openai_configured", lambda: False)
    monkeypatch.setattr("app.main.gemini_configured", lambda: False)

    r = client.post("/ask?provider=auto", json={"prompt": "hi"})
    assert r.status_code == 503
    # CORREÇÃO: O erro final na cadeia de fallback é o do último provedor (gemini)
    assert "gemini_api_key não configurada" in r.json()["detail"].lower()
```

## [29] tests/test_metrics.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **21**
- SHA-256: `856e3ec90817539595c20dfa86ad6eaf05a449d356fc23214aa655621d470784`

```python
# tests/test_metrics.py
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_metrics_exposes_ask_counters():
    # 1) gera uma chamada de sucesso
    r = client.post("/ask?provider=echo", json={"prompt": "ping"})
    assert r.status_code == 200

    # 2) lê /metrics e checa nosso contador customizado
    m = client.get("/metrics")
    assert m.status_code == 200
    text = m.text

    # Deve ter pelo menos um incremento de sucesso para echo
    # Linha esperada (exemplo):
    # ask_requests_total{provider="echo",status="success"} 1.0
    assert 'ask_requests_total{provider="echo",status="success"}' in text, f"Contador não encontrado em /metrics:\n{text}"
```

## [30] tests/test_metrics_error_counter.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **38**
- SHA-256: `850ce9f5c170c26e823a27370a8b2374b6f755dc6ba6af0740ac3da54b92c58a`

```python
# tests/test_metrics_error_counter.py
import re

from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)

_METRIC_NAME = r"ask_requests_total"
# Ex.: ask_requests_total{provider="openai",status="error"} 3
_PATTERN = re.compile(
    rf'^{_METRIC_NAME}\{{provider="openai",status="error"\}}\s+([0-9]+(?:\.[0-9]+)?)\s*$'
)

def _scrape_error_counter() -> float:
    r = client.get("/metrics")
    assert r.status_code == 200
    for line in r.text.splitlines():
        m = _PATTERN.match(line.strip())
        if m:
            return float(m.group(1))
    # Se nunca apareceu, considere zero
    return 0.0


def test_error_counter_increments_on_openai_503():
    # 1) Lê valor atual do contador de erros do openai
    before = _scrape_error_counter()

    # 2) Dispara um 503: provider=openai sem OPENAI_API_KEY
    payload = {"prompt": "ping"}
    r = client.post("/ask?provider=openai", json=payload)
    assert r.status_code == 503

    # 3) Lê novamente e valida incremento de +1
    after = _scrape_error_counter()
    assert after == before + 1, f"Esperava {before}+1, obtive {after}"
```

## [31] tests/test_observability.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **34**
- SHA-256: `53d2b047fd810fb276cdb3d760cc915c7f70e908c6a5dc9dbe730d7a21d0b145`

```python
# tests/test_observability.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.testclient import TestClient

from app.main import app
from app.observability import RequestIDMiddleware, TraceMiddleware, logger

client = TestClient(app)


def test_logger_is_configured_and_usable():
    # Deve existir e ter método .info (logger estruturado configurado)
    assert logger is not None
    assert hasattr(logger, "info")


def test_middlewares_exist_and_are_valid_classes():
    # Ambas as classes devem ser middlewares Starlette válidos
    assert issubclass(TraceMiddleware, BaseHTTPMiddleware)
    assert issubclass(RequestIDMiddleware, BaseHTTPMiddleware)


def test_x_request_id_present_on_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert r.headers["X-Request-ID"]  # não vazio


def test_x_request_id_propagation_on_ready():
    custom = "req-observability-123"
    r = client.get("/ready", headers={"x-request-id": custom})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == custom
```

## [32] tests/test_openai_client.py
- Last modified: **2025-09-14 16:58:04**
- Lines: **52**
- SHA-256: `672c43b1e2b6b573e190b75f88338f9d03a7ea2175e51229a2cdef1b8ab5bdc6`

```python
# =============================================================================
# File: tests/test_openai_client.py
# Version: 2025-09-14 17:00:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO DEFINITIVA: A função mock `mock_create` agora aceita o
#   argumento `self` para simular corretamente um método de instância.
# =============================================================================
import pytest
import asyncio
from app import openai_client

class DummyMessage:
    def __init__(self, content):
        self.content = content

class DummyChoice:
    def __init__(self, content):
        self.message = DummyMessage(content)

class DummyUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15

class DummyResp:
    def __init__(self, text):
        self.choices = [DummyChoice(text)]
        self.usage = DummyUsage()

@pytest.mark.asyncio
async def test_ask_openai_mock(monkeypatch):
    monkeypatch.setattr(openai_client.settings, "OPENAI_API_KEY", "dummy-key-for-test")

    # CORREÇÃO: A função de mock precisa aceitar `self` como primeiro argumento
    # para simular corretamente um método de uma instância de classe.
    async def mock_create(self, **kwargs):
        await asyncio.sleep(0) # simula I/O
        return DummyResp("Paris")

    monkeypatch.setattr(
        "openai.resources.chat.completions.AsyncCompletions.create",
        mock_create
    )

    result = await openai_client.ask_openai("Qual a capital da França?")

    assert result["provider"] == "openai"
    assert result["model"] == openai_client.settings.OPENAI_MODEL
    assert result["answer"] == "Paris"
    assert result["usage"]["total_tokens"] == 15
```

## [33] tests/test_request_id.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **20**
- SHA-256: `2ebc4c965dc3223c112ce5d370ddd116ed9389db792237d8c1574becffef2452`

```python
# tests/test_request_id.py
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_response_contains_request_id_header():
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert r.headers["X-Request-ID"]  # não vazio


def test_request_id_is_propagated_from_request_to_response():
    custom = "abc123xyz"
    r = client.get("/ready", headers={"x-request-id": custom})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == custom
```

## [34] tests/test_request_id_header.py
- Last modified: **2025-09-14 11:44:44**
- Lines: **16**
- SHA-256: `f96698342a6fa9152826a1d9e66d5acba5590c19ed5735b0137ddc68bc89db2b`

```python
# tests/test_request_id_header.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_response_contains_request_id_header():
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers

def test_request_id_is_propagated_from_request_to_response():
    custom = "abc123xyz"
    r = client.get("/ready", headers={"X-Request-ID": custom})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == custom
```

## [35] tools/guard_update.py
- Last modified: **2025-09-14 12:41:05**
- Lines: **134**
- SHA-256: `ad8e6b85912848227fe0d097e22e08ed8813334959883fd6709ff969f350b738`

```python
#!/usr/bin/env python3
# =============================================================================
# File: tools/guard_update.py
# Version: 2025-09-14 13:45:00 -03 (America/Sao_Paulo)
# Purpose:
#   Garante que uma NOVA versão de arquivo (temp) não "encolha" nem perca
#   funções/classes públicas da versão ATUAL. Uso principal:
#   - Pre-commit/CI antes de aceitar substituição integral.
# Rules:
#   - NOVO.num_linhas >= ATUAL.num_linhas  (a menos que --allow-shrink)
#   - NOVO mantém TODAS funções/classes "públicas" (sem prefixo "_")
#   - NOVO preserva o bloco de header (primeiros comentários consecutivos)
# Exit codes:
#   0 = ok ; 1 = erro de regra ; 2 = erro de execução (I/O, parsing, etc.)
# =============================================================================

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Set, Tuple


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def header_block(text: str) -> str:
    lines = text.splitlines()
    hdr = []
    for ln in lines:
        if ln.strip().startswith("#"):
            hdr.append(ln)
        else:
            break
    return "\n".join(hdr).strip()


def public_symbols_py(text: str) -> Tuple[Set[str], Set[str]]:
    """
    Retorna (funções_públicas, classes_públicas) a partir do AST.
    Definição de "público": nome não inicia com "_".
    """
    funcs: Set[str] = set()
    klass: Set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return funcs, klass

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if not name.startswith("_"):
                funcs.add(name)
        elif isinstance(node, ast.AsyncFunctionDef):
            name = node.name
            if not name.startswith("_"):
                funcs.add(name)
        elif isinstance(node, ast.ClassDef):
            name = node.name
            if not name.startswith("_"):
                klass.add(name)
    return funcs, klass


def main() -> int:
    ap = argparse.ArgumentParser(description="Guarda contra encolhimento/perda de símbolos ao substituir arquivo.")
    ap.add_argument("--current", required=True, help="Caminho do arquivo ATUAL (no repo).")
    ap.add_argument("--new", required=True, help="Caminho do arquivo NOVO (temp).")
    ap.add_argument("--allow-shrink", dest="allow_shrink", action="store_true",
                    help="Permite diminuir linhas (NÃO recomendado).")
    ap.add_argument("--lang", default="py", choices=["py", "any"], help="Heurística de símbolos (py=AST).")
    args = ap.parse_args()

    cur = Path(args.current)
    new = Path(args.new)

    if not cur.exists():
        print(f"[guard] current file not found: {cur}", file=sys.stderr)
        return 2
    if not new.exists():
        print(f"[guard] new file not found: {new}", file=sys.stderr)
        return 2

    cur_text = read_text(cur)
    new_text = read_text(new)

    cur_lines = cur_text.count("\n") + (0 if cur_text.endswith("\n") else 1)
    new_lines = new_text.count("\n") + (0 if new_text.endswith("\n") else 1)

    # Regra 1: não encolher (corrigido: usar args.allow_shrink)
    if not args.allow_shrink and new_lines < cur_lines:
        print(f"[guard][ERROR] line count decreased: {cur_lines} -> {new_lines} in {cur}", file=sys.stderr)
        return 1

    # Regra 2: manter header
    cur_hdr = header_block(cur_text)
    new_hdr = header_block(new_text)
    if cur_hdr and cur_hdr not in new_text:
        print(f"[guard][ERROR] header block missing or altered in {new}", file=sys.stderr)
        return 1

    # Regra 3: manter símbolos públicos (para .py)
    if args.lang == "py":
        cur_funcs, cur_classes = public_symbols_py(cur_text)
        new_funcs, new_classes = public_symbols_py(new_text)

        missing_funcs = sorted(cur_funcs - new_funcs)
        missing_classes = sorted(cur_classes - new_classes)

        if missing_funcs or missing_classes:
            if missing_funcs:
                print(f"[guard][ERROR] missing public functions in {new}: {', '.join(missing_funcs)}", file=sys.stderr)
            if missing_classes:
                print(f"[guard][ERROR] missing public classes in {new}: {', '.join(missing_classes)}", file=sys.stderr)
            return 1

    print(f"[guard][OK] {cur} -> {new} (lines {cur_lines}->{new_lines})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[aborted] interrupted by user", file=sys.stderr)
        raise
```

## [36] tools/snapshot_configs.py
- Last modified: **2025-09-14 12:04:40**
- Lines: **290**
- SHA-256: `6407d9d7f76f4bdf2e7626bb17061d46e5b91620495b879e277f48dba4aba0c2`

```python
#!/usr/bin/env python3
# =============================================================================
# File: tools/snapshot_configs.py
# Version: 2025-09-14 12:22:00 -03 (America/Sao_Paulo)
# Purpose:
#   Gera um snapshot consolidado dos arquivos de configuração do projeto
#   (com índice, metadados e conteúdo), para evitar perda de contexto e
#   detectar regressões/acidentes onde partes do arquivo somem.
#
# Uso:
#   python tools/snapshot_configs.py
#   python tools/snapshot_configs.py --output CONFIG_SNAPSHOT.md --manifest CONFIG_SNAPSHOT.manifest.json
#   python tools/snapshot_configs.py --include-ext .py .env .yaml .yml .toml .json .ini .cfg .conf --exclude-dirs app/modules
#
# Saídas padrão:
#   - CONFIG_SNAPSHOT.md               (consolidado com índice + conteúdo)
#   - CONFIG_SNAPSHOT.manifest.json    (manifesto com hash/nº linhas/mtime)
# =============================================================================

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_OUTPUT = "CONFIG_SNAPSHOT.md"
DEFAULT_MANIFEST = "CONFIG_SNAPSHOT.manifest.json"

# extensões típicas de **configuração**
DEFAULT_INCLUDE_EXT = [
    ".py",       # settings.py, config.py etc (código-config)
    ".env", ".env.example",
    ".yaml", ".yml",
    ".toml",
    ".json",
    ".ini", ".cfg", ".conf",
    ".service",           # systemd
    ".properties",
    ".editorconfig",
]

# nomes de arquivos sem extensão que geralmente são config
DEFAULT_INCLUDE_NAMES = [
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
    "Pipfile",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    ".gitattributes",
    ".pre-commit-config.yaml",
    ".prettierrc",
    ".eslintrc", ".eslintrc.json", ".eslintrc.yml",
]

# diretórios a **excluir** (de módulos/artefatos/temporários)
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".vscode",
    ".idea",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "site-packages",
    "venv",
    ".venv",
    "env",
    ".env",      # dir
    ".cache",
    ".ruff_cache",
    ".tox",
    ".coverage",
}

# paths completos a excluir (ajuste se precisar)
DEFAULT_EXCLUDE_PATHS = set()


def is_config_file(path: Path, include_ext: List[str], include_names: List[str]) -> bool:
    if not path.is_file():
        return False
    name = path.name
    suffix = path.suffix.lower()

    # match por nome inteiro
    if name in include_names:
        return True

    # match por extensão (case-insensitive)
    if suffix in [e.lower() for e in include_ext]:
        return True

    # arquivos .env.* (ex: .env.local)
    if name.startswith(".env."):
        return True

    return False


def iter_files(root: Path, exclude_dirs: Iterable[str]) -> Iterable[Path]:
    exclude_dirs_lower = {d.lower() for d in exclude_dirs}
    for dirpath, dirnames, filenames in os.walk(root):
        # prune diretórios
        dirnames[:] = [d for d in dirnames if d.lower() not in exclude_dirs_lower]
        # yield files
        for fn in filenames:
            yield Path(dirpath) / fn


def mtime_str(p: Path) -> str:
    ts = p.stat().st_mtime
    # timezone local do sistema (São Paulo no teu caso)
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text_safely(p: Path) -> Tuple[str, int, str]:
    """
    Lê arquivo como texto (utf-8). Se falhar, tenta latin-1.
    Retorna: (conteudo, num_linhas, hash_sha256_hex)
    """
    raw: bytes
    try:
        raw = p.read_bytes()
    except Exception as e:
        # arquivos especiais podem falhar; retorna vazio para não quebrar
        return f"<<erro ao ler bytes: {e}>>", 0, sha256_of_bytes(b"")

    text: str
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            text = None  # type: ignore
    if text is None:
        # como último recurso, representação binária curta
        head = raw[:256]
        text = f"<<binário ({len(raw)} bytes). head: {head!r}>>"

    line_count = text.count("\n") + (0 if text.endswith("\n") else 1)
    return text, line_count, sha256_of_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera snapshot consolidado de arquivos de configuração.")
    parser.add_argument("--root", default=".", help="Diretório raiz do projeto (default: .)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Arquivo de saída (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help=f"Manifesto JSON (default: {DEFAULT_MANIFEST})")
    parser.add_argument("--include-ext", nargs="*", default=DEFAULT_INCLUDE_EXT,
                        help="Extensões a incluir (ex.: .py .yml .toml ...)")
    parser.add_argument("--include-names", nargs="*", default=DEFAULT_INCLUDE_NAMES,
                        help="Nomes exatos de arquivos a incluir (ex.: Dockerfile Makefile ...)")
    parser.add_argument("--exclude-dirs", nargs="*", default=list(DEFAULT_EXCLUDE_DIRS),
                        help="Diretórios a excluir (nomes, não paths)")
    parser.add_argument("--exclude-paths", nargs="*", default=list(DEFAULT_EXCLUDE_PATHS),
                        help="Paths completos a excluir (começando na raiz). Ex.: app/modules configs/secrets")
    parser.add_argument("--max-bytes", type=int, default=0,
                        help="Se >0, trunca conteúdo por arquivo a este limite (segurança).")
    parser.add_argument("--sort", choices=["path", "mtime"], default="path",
                        help="Ordenação das seções: por path (default) ou mtime.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_md = Path(args.output).resolve()
    out_manifest = Path(args.manifest).resolve()

    # normaliza exclude paths
    exclude_paths_abs = { (root / p).resolve() for p in args.exclude_paths }

    candidates: List[Path] = []
    for p in iter_files(root, args.exclude_dirs):
        if any(str(p.resolve()).startswith(str(ex)) for ex in exclude_paths_abs):
            continue
        if is_config_file(p, args.include_ext, args.include_names):
            candidates.append(p)

    # ordenação
    if args.sort == "mtime":
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=False)
    else:
        candidates.sort(key=lambda p: str(p).lower())

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %z") or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entries = []
    total_lines = 0

    for p in candidates:
        content, line_count, sha = read_text_safely(p)
        if args.max_bytes and len(content.encode("utf-8", "ignore")) > args.max_bytes:
            # truncar mantendo info
            encoded = content.encode("utf-8", "ignore")
            content = encoded[: args.max_bytes].decode("utf-8", "ignore") + "\n<<TRUNCATED>>\n"
        entries.append({
            "path": str(p.relative_to(root)),
            "mtime": mtime_str(p),
            "lines": line_count,
            "sha256": sha,
            "content": content,
        })
        total_lines += line_count

    # manifesto JSON (para CI/automatização)
    manifest = {
        "generated_at": now,
        "root": str(root),
        "file_count": len(entries),
        "total_lines": total_lines,
        "hash_algorithm": "sha256",
        "files": [
            {k: v for k, v in e.items() if k != "content"} for e in entries
        ],
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # MD consolidado com índice
    lines: List[str] = []
    lines.append("# CONFIG SNAPSHOT\n")
    lines.append(f"- Generated at: **{now}**\n")
    lines.append(f"- Root: `{root}`\n")
    lines.append(f"- Files: **{len(entries)}**\n")
    lines.append(f"- Total config lines: **{total_lines}**\n")
    lines.append("\n---\n")
    lines.append("## Index\n")
    if entries:
        width = len(str(len(entries)))
    else:
        width = 1
    for i, e in enumerate(entries, start=1):
        idx = str(i).rjust(width)
        lines.append(f"- [{idx}] `{e['path']}` — {e['lines']} lines — mtime {e['mtime']} — sha256 `{e['sha256'][:12]}…`")
    lines.append("\n---\n")

    for i, e in enumerate(entries, start=1):
        lines.append(f"## [{i}] {e['path']}")
        lines.append(f"- Last modified: **{e['mtime']}**")
        lines.append(f"- Lines: **{e['lines']}**")
        lines.append(f"- SHA-256: `{e['sha256']}`\n")
        # escolhe linguagem do bloco de código
        code_lang = ""
        suffix = Path(e["path"]).suffix.lower()
        if suffix in (".py",):
            code_lang = "python"
        elif suffix in (".yml", ".yaml"):
            code_lang = "yaml"
        elif suffix in (".json",):
            code_lang = "json"
        elif suffix in (".toml",):
            code_lang = "toml"
        elif suffix in (".ini", ".cfg", ".conf", ".properties"):
            code_lang = ""
        elif e["path"].endswith("Dockerfile"):
            code_lang = "dockerfile"
        elif e["path"].endswith("Makefile"):
            code_lang = "makefile"

        lines.append(f"```{code_lang}")
        lines.append(e["content"].rstrip("\n"))
        lines.append("```\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] snapshot: {out_md}")
    print(f"[ok] manifest: {out_manifest}")
    print(f"[info] files: {len(entries)} | lines: {total_lines}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[aborted] interrupted by user", file=sys.stderr)
        raise
```

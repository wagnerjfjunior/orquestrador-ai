# CONFIG SNAPSHOT

- Generated at: **2025-09-14 12:11:51 **

- Root: `/Users/wagnerjfjunior/orquestrador-ai`

- Files: **34**

- Total config lines: **2012**


---

## Index

- [ 1] `.env.example` — 29 lines — mtime 2025-09-13 07:55:28 — sha256 `274aceefe36c…`
- [ 2] `.github/workflows/ci.yml` — 39 lines — mtime 2025-09-13 19:22:11 — sha256 `1741b400f725…`
- [ 3] `.gitignore` — 10 lines — mtime 2025-09-12 01:56:56 — sha256 `b746016476e6…`
- [ 4] `app/__init__.py` — 2 lines — mtime 2025-09-13 15:30:17 — sha256 `f0fb5e1d3cbe…`
- [ 5] `app/cache.py` — 34 lines — mtime 2025-09-13 18:33:31 — sha256 `0a8edfe8a856…`
- [ 6] `app/config.py` — 62 lines — mtime 2025-09-13 18:33:31 — sha256 `ffa9c871ab06…`
- [ 7] `app/gemini_client.py` — 93 lines — mtime 2025-09-13 18:33:31 — sha256 `d5e2e34bf8cf…`
- [ 8] `app/judge.py` — 71 lines — mtime 2025-09-14 11:27:38 — sha256 `874ade3ba271…`
- [ 9] `app/main.py` — 239 lines — mtime 2025-09-14 11:40:33 — sha256 `e77349199271…`
- [10] `app/metrics.py` — 44 lines — mtime 2025-09-13 18:33:31 — sha256 `672d96010d9c…`
- [11] `app/observability.py` — 111 lines — mtime 2025-09-14 11:32:26 — sha256 `5a5528e1c8ab…`
- [12] `app/openai_client.py` — 128 lines — mtime 2025-09-13 18:33:31 — sha256 `0aa4158cb39c…`
- [13] `app/retry.py` — 42 lines — mtime 2025-09-13 18:33:31 — sha256 `f7539c2a2673…`
- [14] `app/utils/__init__.py` — 2 lines — mtime 2025-09-13 15:31:19 — sha256 `f0fb5e1d3cbe…`
- [15] `app/utils/retry.py` — 42 lines — mtime 2025-09-13 18:33:31 — sha256 `d22081dab42b…`
- [16] `CONFIG_SNAPSHOT.manifest.json` — 207 lines — mtime 2025-09-14 12:07:57 — sha256 `d4f8692cf60a…`
- [17] `cy.yml` — 63 lines — mtime 2025-09-13 15:38:40 — sha256 `9daf10926641…`
- [18] `Dockerfile` — 27 lines — mtime 2025-09-13 16:37:38 — sha256 `5dc22d6e9ce9…`
- [19] `Makefile` — 42 lines — mtime 2025-09-13 16:38:26 — sha256 `2b0fac71f475…`
- [20] `pyproject.toml` — 22 lines — mtime 2025-09-12 23:43:45 — sha256 `d8c205569aa1…`
- [21] `render.yaml` — 16 lines — mtime 2025-09-13 17:51:23 — sha256 `ea92f0fb004c…`
- [22] `ruff.toml` — 19 lines — mtime 2025-09-13 18:38:03 — sha256 `13e62274b996…`
- [23] `tests/test_ask_providers.py` — 78 lines — mtime 2025-09-13 18:33:31 — sha256 `a894b30cfd10…`
- [24] `tests/test_basic.py` — 22 lines — mtime 2025-09-13 18:33:31 — sha256 `552f7a87700f…`
- [25] `tests/test_duel_no_providers.py` — 18 lines — mtime 2025-09-14 11:42:30 — sha256 `dbbf7419d363…`
- [26] `tests/test_duel_openai_only.py` — 25 lines — mtime 2025-09-14 11:43:21 — sha256 `440c8d4b1b3b…`
- [27] `tests/test_fallback.py` — 53 lines — mtime 2025-09-13 18:33:31 — sha256 `5cc63c636101…`
- [28] `tests/test_metrics.py` — 21 lines — mtime 2025-09-13 18:33:31 — sha256 `856e3ec90817…`
- [29] `tests/test_metrics_error_counter.py` — 38 lines — mtime 2025-09-13 18:33:31 — sha256 `850ce9f5c170…`
- [30] `tests/test_observability.py` — 34 lines — mtime 2025-09-13 18:33:31 — sha256 `53d2b047fd81…`
- [31] `tests/test_openai_client.py` — 53 lines — mtime 2025-09-13 18:33:31 — sha256 `a15c6d5bdb31…`
- [32] `tests/test_request_id.py` — 20 lines — mtime 2025-09-13 18:33:31 — sha256 `2ebc4c965dc3…`
- [33] `tests/test_request_id_header.py` — 16 lines — mtime 2025-09-14 11:44:44 — sha256 `f96698342a6f…`
- [34] `tools/snapshot_configs.py` — 290 lines — mtime 2025-09-14 12:04:40 — sha256 `6407d9d7f76f…`

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

## [3] .gitignore
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

## [4] app/__init__.py
- Last modified: **2025-09-13 15:30:17**
- Lines: **2**
- SHA-256: `f0fb5e1d3cbe63ad8149256a91c4b7228cbedfca932ffc0d9cb6086adee6c92f`

```python
# app/utils/__init__.py
# Torna 'utils' um pacote Python.
```

## [5] app/cache.py
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

## [6] app/config.py
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

## [7] app/gemini_client.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **93**
- SHA-256: `d5e2e34bf8cf3534cb68e468bd91adbcb8d6212d8b57307b31cf5a23d5f232ed`

```python
# app/gemini_client.py
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


def ask_gemini(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,  # mantido para simetria
    **extra: Any,
) -> Dict[str, Any]:
    """
    Envia um prompt ao Gemini e retorna resposta normalizada.

    Parâmetros:
      - prompt: texto do usuário
      - model: override do modelo (default: settings.GEMINI_MODEL)
      - timeout: mantido por simetria (a SDK atual não aceita timeout direto)
      - **extra: futuros parâmetros (ex.: generation_config)

    Retorno:
      {
        "provider": "gemini",
        "model": "<modelo>",
        "answer": "<texto>",
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
      }
    """
    mdl = model or settings.GEMINI_MODEL
    tmo = timeout or settings.PROVIDER_TIMEOUT

    logger.info("provider.gemini.request", model=mdl, timeout=tmo)

    try:
        gmodel = _build_model(mdl)

        # Permite passar configs adicionais (ex: generation_config={"temperature":0.3})
        generation_config = extra.get("generation_config")
        safety_settings = extra.get("safety_settings")

        resp = gmodel.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # A SDK expõe texto final em resp.text
        content = (getattr(resp, "text", "") or "").strip()

        # A SDK atual não retorna usage padronizado — deixamos None
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

        logger.info("provider.gemini.success", model=mdl)

        return {
            "provider": "gemini",
            "model": mdl,
            "answer": content,
            "usage": usage,
        }

    except genai.types.generation_types.BlockedPromptException as e:  # type: ignore[attr-defined]
        logger.info("provider.gemini.blocked_prompt", error=str(e))
        raise RuntimeError("Prompt bloqueado pela política do Gemini.") from e
    except genai.types.generation_types.StopCandidateException as e:  # type: ignore[attr-defined]
        logger.info("provider.gemini.stop_candidate", error=str(e))
        raise RuntimeError("Geração interrompida pelo Gemini.") from e
    except Exception as e:
        logger.info("provider.gemini.unexpected_error", error=str(e))
        raise RuntimeError("Erro inesperado ao chamar o Gemini.") from e
```

## [8] app/judge.py
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

## [9] app/main.py
- Last modified: **2025-09-14 11:40:33**
- Lines: **239**
- SHA-256: `e7734919927199deec1f1a0ff82840a8347034f4eea8c7ffa6ff39b236fa1745`

```python
# app/main.py
from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# 1) Métricas primeiro (mais interno)
setup_metrics(app)
# 2) Tracing no meio
app.add_middleware(TraceMiddleware)
# 3) RequestID MAIS externo (último a escrever na resposta)
app.add_middleware(RequestIDMiddleware)

# --- Rotas básicas ---
@app.get("/", tags=["infra"])
def root():
    logger.info("root.live")
    return {"status": "live"}

@app.get("/health", tags=["infra"])
def health():
    logger.info("health.ok")
    return {"status": "ok"}

@app.get("/ready", tags=["infra"])
def readiness():
    logger.info("readiness.ok")
    return {"status": "ready"}


# ----------------- Providers (mantém comportamento do original) -----------------
def _provider_is_configured(name: str) -> bool:
    name = (name or "").lower()
    if name == "openai":
        return openai_configured()
    if name == "gemini":
        return gemini_configured()
    if name == "echo":
        return True
    return False

def _provider_call(name: str, prompt: str) -> Dict[str, Any]:
    name = (name or "").lower()

    if name == "echo":
        logger.info("ask.echo", prompt=prompt)
        return {"provider": "echo", "answer": prompt, "output": prompt}

    if name == "openai":
        if not openai_configured():
            # mensagem específica esperada nos testes
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY não configurada.")
        return ask_openai(prompt)

    if name == "gemini":
        if not gemini_configured():
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY não configurada.")
        return ask_gemini(prompt)

    raise HTTPException(status_code=400, detail=f"Provider não suportado: {name}")

def _fallback_chain(primary: str | None) -> List[str]:
    """
    - provider='auto' ou None  -> usar cadeia do settings.PROVIDER_FALLBACK
    - provider explícito       -> tentar somente esse provider (sem fallback)
    """
    if not primary or primary.lower() == "auto":
        return [p.lower() for p in settings.PROVIDER_FALLBACK]
    return [primary.lower()]

# ----------------- DUEL (novo, sem quebrar nada do original) -----------------
def _try_call(p: str, prompt: str) -> Tuple[str, Dict[str, Any] | None, str | None]:
    try:
        if not _provider_is_configured(p):
            return p, None, "não configurado"
        return p, _provider_call(p, prompt), None
    except HTTPException as http_exc:
        return p, None, f"http_{http_exc.status_code}: {http_exc.detail}"
    except Exception as e:
        return p, None, f"erro: {e}"

def _ask_duel(prompt: str) -> Dict[str, Any]:
    # Executa OpenAI e Gemini em paralelo para reduzir latência
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_try_call, "openai", prompt): "openai",
            ex.submit(_try_call, "gemini", prompt): "gemini",
        }
        results: Dict[str, Dict[str, Any] | None] = {"openai": None, "gemini": None}
        errors: Dict[str, str | None] = {"openai": None, "gemini": None}
        for fut in as_completed(futures):
            prov = futures[fut]
            _, resp, err = fut.result()
            results[prov] = resp
            errors[prov] = err

    # Juiz por LLM (preferência) com fallback heurístico (dentro de judge_answers)
    verdict_llm = judge_answers(
        prompt,
        (results["openai"] or {}).get("answer") or "",
        (results["gemini"] or {}).get("answer") or "",
    )

    if verdict_llm["provider"] == "heuristic" and not (
        (results["openai"] or {}).get("answer") or (results["gemini"] or {}).get("answer")
    ):
        verdict = {"winner": "none", "rationale": "nenhum provider retornou conteúdo"}
    else:
        map_winner = {"a": "openai", "b": "gemini", "tie": "tie"}
        verdict = {
            "winner": map_winner.get(verdict_llm.get("winner"), "tie"),
            "rationale": verdict_llm.get("reason"),
        }

    return {
        "mode": "duel",
        "prompt": prompt,
        "responses": {
            "openai": {
                "ok": results["openai"] is not None,
                "answer": (results["openai"] or {}).get("answer"),
                "error": errors["openai"],
            },
            "gemini": {
                "ok": results["gemini"] is not None,
                "answer": (results["gemini"] or {}).get("answer"),
                "error": errors["gemini"],
            },
        },
        "verdict": verdict,
    }

# ----------------- Rotas -----------------
@app.post("/ask", tags=["ask"])
def ask(provider: str = "echo", payload: dict = Body(...), use_fallback: bool = True, mode: str | None = None):
    """
    - provider: echo | openai | gemini | auto | duel  (duel é atalho de mode=duel)
    - use_fallback: só tem efeito quando provider=auto (explícito ignora fallback)
    - mode=duel: roda OpenAI e Gemini em paralelo, devolve as duas respostas + veredito
    """
    prompt = payload.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório no corpo JSON.")

    # Novo modo duelo também por /ask
    if (mode or "").lower() == "duel" or (provider or "").lower() == "duel":
        logger.info("ask.duel.start")
        result = _ask_duel(prompt)
        logger.info("ask.duel.end", verdict=result.get("verdict", {}).get("winner"))
        return result

    chain = _fallback_chain(provider)
    is_auto = (provider or "").lower() == "auto"

    start = time.perf_counter()
    last_error: Exception | None = None

    # Provider explícito: sem fallback, registra métricas direto (mantém original)
    if not is_auto:
        p = chain[0]
        try:
            resp = _provider_call(p, prompt)
            duration_ms = (time.perf_counter() - start) * 1000
            record_ask(p, "success", duration_ms)
            return resp
        except HTTPException as http_exc:
            duration_ms = (time.perf_counter() - start) * 1000
            record_ask(p, "error", duration_ms)
            logger.info("ask.http_exception", provider=p, status=http_exc.status_code)
            raise http_exc
        except RuntimeError as runtime_err:
            duration_ms = (time.perf_counter() - start) * 1000
            record_ask(p, "error", duration_ms)
            logger.info("ask.provider_runtime_error", provider=p, error=str(runtime_err))
            raise HTTPException(status_code=502, detail=str(runtime_err))

    # provider=auto: tenta cadeia com fallback (mantém original)
    for idx, p in enumerate(chain):
        try:
            if not use_fallback and idx > 0:
                break

            if not _provider_is_configured(p):
                logger.info("ask.provider_not_configured", provider=p)
                last_error = HTTPException(status_code=503, detail=f"Provider não configurado: {p}")
                continue

            resp = _provider_call(p, prompt)
            duration_ms = (time.perf_counter() - start) * 1000
            record_ask(p, "success", duration_ms)
            logger.info("ask.provider_success", provider=p)
            return resp

        except HTTPException as http_exc:
            duration_ms = (time.perf_counter() - start) * 1000
            record_ask(p, "error", duration_ms)
            logger.info("ask.http_exception", provider=p, status=http_exc.status_code)
            last_error = http_exc
            if not use_fallback:
                raise http_exc
        except RuntimeError as runtime_err:
            duration_ms = (time.perf_counter() - start) * 1000
            record_ask(p, "error", duration_ms)
            logger.info("ask.provider_runtime_error", provider=p, error=str(runtime_err))
            last_error = runtime_err
            if not use_fallback:
                raise HTTPException(status_code=502, detail=str(runtime_err))

    # auto e todos falharam
    if isinstance(last_error, HTTPException):
        raise last_error
    detail = str(last_error) if last_error else "Falha ao atender requisição em todos os provedores."
    raise HTTPException(status_code=502, detail=detail)

@app.post("/duel", tags=["duel"])
def duel(payload: dict = Body(...)):
    """
    Roda o modo duelo explicitamente (atalho para /ask com mode=duel).
    """
    prompt = payload.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório no corpo JSON.")
    return _ask_duel(prompt)
```

## [10] app/metrics.py
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

## [11] app/observability.py
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

## [12] app/openai_client.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **128**
- SHA-256: `0aa4158cb39c9fb4c2bc841891b8177a552bb21b153dc6222405c23e4d27a5cd`

```python
# app/openai_client.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError

from app.config import settings
from app.observability import logger
from app.utils.retry import RetryExceededError, retry


def is_configured() -> bool:
    """
    Retorna True se houver OPENAI_API_KEY configurada.
    """
    return bool(settings.OPENAI_API_KEY)


def _build_client(timeout: Optional[float] = None) -> OpenAI:
    """
    Constroi o cliente OpenAI com a API key do settings.
    Permite sobrescrever timeout por chamada.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não configurada.")

    # openai-python v1.x
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    # Ajuste de timeout por requisição (create(..., timeout=...))
    # Mantemos simples aqui; se quiser timeout global, dá pra ajustar http_client no client.
    return client


def ask_openai(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Envia um prompt para o OpenAI Chat Completions e retorna resposta normalizada.

    Parâmetros:
      - prompt: texto do usuário
      - model: override do modelo (default: settings.OPENAI_MODEL)
      - timeout: timeout em segundos para esta chamada (default: settings.PROVIDER_TIMEOUT)
      - **extra: espaço para parâmetros futuros (temperature, top_p, etc.)
                aceita também _sleep (Callable[[float], None]) usado nos testes de retry

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

    # Permite testes injetarem um sleep no-op sem mudar a assinatura pública
    _sleep: Optional[Callable[[float], None]] = extra.pop("_sleep", None)

    client = _build_client(timeout=tmo)
    logger.info("provider.openai.request", model=mdl)

    def _do_call() -> Dict[str, Any]:
        # API Chat Completions (modelos como gpt-4o-mini)
        resp = client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            timeout=tmo,
            **extra,
        )

        content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

        logger.info(
            "provider.openai.success",
            model=mdl,
            total_tokens=usage["total_tokens"],
        )

        return {
            "provider": "openai",
            "model": mdl,
            "answer": content,
            "usage": usage,
        }

    try:
        # Retry leve apenas para erros transitórios de rede
        return retry(
            _do_call,
            retries=2,            # até 2 tentativas adicionais (total máx = 3)
            backoff_ms=200,       # exponencial simples: 200ms, depois 400ms
            retry_on=(APIConnectionError,),
            sleep=_sleep,         # nos testes, injetamos no-op para não atrasar
        )

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
        # Pode ser lançado antes do retry ou mesmo após esgotar tentativas em algum caminho
        logger.info("provider.openai.connection_error", error=str(e))
        raise RuntimeError("Erro de conexão com a OpenAI.") from e
    except RetryExceededError as e:
        logger.info("provider.openai.retry_exceeded", error=str(e))
        raise RuntimeError("Erro de conexão com a OpenAI (tentativas esgotadas).") from e
    except Exception as e:
        logger.info("provider.openai.unexpected_error", error=str(e))
        raise RuntimeError("Erro inesperado ao chamar a OpenAI.") from e
```

## [13] app/retry.py
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

## [14] app/utils/__init__.py
- Last modified: **2025-09-13 15:31:19**
- Lines: **2**
- SHA-256: `f0fb5e1d3cbe63ad8149256a91c4b7228cbedfca932ffc0d9cb6086adee6c92f`

```python
# app/utils/__init__.py
# Torna 'utils' um pacote Python.
```

## [15] app/utils/retry.py
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

## [16] CONFIG_SNAPSHOT.manifest.json
- Last modified: **2025-09-14 12:07:57**
- Lines: **207**
- SHA-256: `d4f8692cf60a54f109135fecac59a79a29770887efaacd9e3b7c4fb7f831765c`

```json
{
  "generated_at": "2025-09-14 12:07:57 ",
  "root": "/Users/wagnerjfjunior/orquestrador-ai",
  "file_count": 33,
  "total_lines": 1805,
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
      "mtime": "2025-09-13 18:33:31",
      "lines": 93,
      "sha256": "d5e2e34bf8cf3534cb68e468bd91adbcb8d6212d8b57307b31cf5a23d5f232ed"
    },
    {
      "path": "app/judge.py",
      "mtime": "2025-09-14 11:27:38",
      "lines": 71,
      "sha256": "874ade3ba27169a304737fa60913fdb1e3068b4328a6d87c0ab1ee791c19794d"
    },
    {
      "path": "app/main.py",
      "mtime": "2025-09-14 11:40:33",
      "lines": 239,
      "sha256": "e7734919927199deec1f1a0ff82840a8347034f4eea8c7ffa6ff39b236fa1745"
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
      "mtime": "2025-09-13 18:33:31",
      "lines": 128,
      "sha256": "0aa4158cb39c9fb4c2bc841891b8177a552bb21b153dc6222405c23e4d27a5cd"
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
      "mtime": "2025-09-13 18:33:31",
      "lines": 78,
      "sha256": "a894b30cfd10bfaf9e99a5650d528919007f253297fc6aaebafe3adb0108e4bc"
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
      "mtime": "2025-09-14 11:43:21",
      "lines": 25,
      "sha256": "440c8d4b1b3baf3ce17815d30dfd5cdf086d8367bd7ea585ff84db0b092af149"
    },
    {
      "path": "tests/test_fallback.py",
      "mtime": "2025-09-13 18:33:31",
      "lines": 53,
      "sha256": "5cc63c636101c4c61d763710421c005964aa921721c03ae2da607406a35c99cd"
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
      "mtime": "2025-09-13 18:33:31",
      "lines": 53,
      "sha256": "a15c6d5bdb31ebb1d5d5da138b90433be4c5a0d46b57b3a57d40fd6c346dd478"
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
      "path": "tools/snapshot_configs.py",
      "mtime": "2025-09-14 12:04:40",
      "lines": 290,
      "sha256": "6407d9d7f76f4bdf2e7626bb17061d46e5b91620495b879e277f48dba4aba0c2"
    }
  ]
}
```

## [17] cy.yml
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

## [18] Dockerfile
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

## [19] Makefile
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

## [20] pyproject.toml
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

## [21] render.yaml
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

## [22] ruff.toml
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

## [23] tests/test_ask_providers.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **78**
- SHA-256: `a894b30cfd10bfaf9e99a5650d528919007f253297fc6aaebafe3adb0108e4bc`

```python
# tests/test_ask_providers.py
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ask_openai_success(monkeypatch):
    # finge que a chave está configurada
    monkeypatch.setattr("app.main.openai_configured", lambda: True)

    # mock da chamada ao cliente openai
    def fake_ask_openai(prompt):
        assert prompt == "olá openai"
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "answer": "oi, daqui é o openai",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr("app.main.ask_openai", fake_ask_openai)

    r = client.post("/ask?provider=openai", json={"prompt": "olá openai"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "openai"
    assert data["answer"] == "oi, daqui é o openai"
    assert "usage" in data


def test_ask_gemini_success(monkeypatch):
    # finge que a chave está configurada
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    # mock da chamada ao cliente gemini
    def fake_ask_gemini(prompt):
        assert prompt == "olá gemini"
        return {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "answer": "oi, daqui é o gemini",
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }

    monkeypatch.setattr("app.main.ask_gemini", fake_ask_gemini)

    r = client.post("/ask?provider=gemini", json={"prompt": "olá gemini"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "gemini"
    assert data["answer"] == "oi, daqui é o gemini"
    assert "usage" in data


def test_ask_openai_not_configured(monkeypatch):
    # força "não configurado" → deve retornar 503
    monkeypatch.setattr("app.main.openai_configured", lambda: False)

    r = client.post("/ask?provider=openai", json={"prompt": "qualquer"})
    assert r.status_code == 503
    assert "não configurada" in r.json()["detail"].lower()


def test_ask_gemini_provider_error(monkeypatch):
    # configurado, mas cliente lança erro → deve virar 502
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    def boom(prompt):
        raise RuntimeError("Rate limit atingido na OpenAI. Tente novamente mais tarde.")  # exemplo de msg

    monkeypatch.setattr("app.main.ask_gemini", lambda prompt: boom(prompt))

    r = client.post("/ask?provider=gemini", json={"prompt": "teste"})
    assert r.status_code == 502
    assert "erro" in r.json()["detail"].lower() or "limit" in r.json()["detail"].lower()
```

## [24] tests/test_basic.py
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

## [25] tests/test_duel_no_providers.py
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

## [26] tests/test_duel_openai_only.py
- Last modified: **2025-09-14 11:43:21**
- Lines: **25**
- SHA-256: `440c8d4b1b3baf3ce17815d30dfd5cdf086d8367bd7ea585ff84db0b092af149`

```python
# tests/test_duel_openai_only.py
from fastapi.testclient import TestClient
import app.main as m

client = TestClient(m.app)

def test_duel_openai_only_ok():
    # Só OpenAI “configurado”
    m.openai_configured = lambda: True
    m.gemini_configured = lambda: False

    # Mock do provider
    def _fake_provider_call(name, prompt):
        assert name == "openai"
        return {"provider": "openai", "answer": "Paris é a capital da França."}

    m._provider_call = _fake_provider_call

    resp = client.post("/duel", json={"prompt": "qual a capital da França?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "duel"
    assert body["responses"]["openai"]["ok"] is True
    assert "Paris" in (body["responses"]["openai"]["answer"] or "")
    assert body["verdict"]["winner"] in ("openai", "tie")
```

## [27] tests/test_fallback.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **53**
- SHA-256: `5cc63c636101c4c61d763710421c005964aa921721c03ae2da607406a35c99cd`

```python
# tests/test_fallback.py
import pytest
from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_fallback_openai_falha_e_gemini_sucesso(monkeypatch):
    # Cadeia padrão no settings costuma ser ["openai", "gemini"] — assumimos isso.
    monkeypatch.setattr("app.main.openai_configured", lambda: True)
    monkeypatch.setattr("app.main.gemini_configured", lambda: True)

    def boom(prompt):
        raise RuntimeError("Erro simulado no OpenAI")

    def ok(prompt):
        return {"provider": "gemini", "model": "gemini-1.5-flash", "answer": "ok gemini", "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}}

    monkeypatch.setattr("app.main.ask_openai", lambda prompt: boom(prompt))
    monkeypatch.setattr("app.main.ask_gemini", lambda prompt: ok(prompt))

    r = client.post("/ask?provider=auto", json={"prompt": "hi"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "gemini"
    assert data["answer"] == "ok gemini"


def test_fallback_provider_explicito_sem_fallback(monkeypatch):
    # Se use_fallback=false, não deve tentar o próximo
    monkeypatch.setattr("app.main.openai_configured", lambda: True)

    def boom(prompt):
        raise RuntimeError("Erro simulado no OpenAI")

    monkeypatch.setattr("app.main.ask_openai", lambda prompt: boom(prompt))

    r = client.post("/ask?provider=openai&use_fallback=false", json={"prompt": "hi"})
    assert r.status_code == 502
    assert "erro" in r.json()["detail"].lower() or "simulado" in r.json()["detail"].lower()


def test_fallback_todos_falham(monkeypatch):
    # Nem openai nem gemini disponíveis (ou ambos falham)
    monkeypatch.setattr("app.main.openai_configured", lambda: False)
    monkeypatch.setattr("app.main.gemini_configured", lambda: False)

    r = client.post("/ask?provider=auto", json={"prompt": "hi"})
    # Como nenhum está configurado, o último erro é 503 (não configurado)
    assert r.status_code == 503
    assert "não configurado" in r.json()["detail"].lower()
```

## [28] tests/test_metrics.py
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

## [29] tests/test_metrics_error_counter.py
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

## [30] tests/test_observability.py
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

## [31] tests/test_openai_client.py
- Last modified: **2025-09-13 18:33:31**
- Lines: **53**
- SHA-256: `a15c6d5bdb31ebb1d5d5da138b90433be4c5a0d46b57b3a57d40fd6c346dd478`

```python
# tests/test_openai_client.py
import pytest

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


def test_ask_openai_mock(monkeypatch):
    """
    Testa ask_openai() sem chamar a API real, mockando a resposta.
    """

    # força settings a ter uma API key dummy
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    # patch no _build_client -> retorna objeto com chat.completions.create
    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return DummyResp("Paris")

    monkeypatch.setattr(openai_client, "_build_client", lambda timeout=None: DummyClient())

    result = openai_client.ask_openai("Qual a capital da França?")

    assert result["provider"] == "openai"
    assert result["model"] == openai_client.settings.OPENAI_MODEL
    assert result["answer"] == "Paris"
    assert result["usage"]["total_tokens"] == 15
```

## [32] tests/test_request_id.py
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

## [33] tests/test_request_id_header.py
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

## [34] tools/snapshot_configs.py
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

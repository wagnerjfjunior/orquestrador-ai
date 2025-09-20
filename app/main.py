# app/main.py
from __future__ import annotations

import asyncio
import os
import time
import uuid
import random
import inspect
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

# ---- Clients locais (mantém compat com sua base) ----
from .openai_client import ask as ask_openai, is_configured as openai_is_configured
from .gemini_client import ask as ask_gemini, is_configured as gemini_is_configured

# Opcional: utilidades do judge (apenas reexport para tests que importam do main)
try:
    from .judge import judge_answers as judge_answers  # os testes monkeypatcham app.main.judge_answers
except Exception:
    # Fallback simples caso o módulo não exponha judge_answers
    async def judge_answers(prompt: str, a: str, b: str) -> Dict[str, str]:  # type: ignore
        la, lb = len(a or ""), len(b or "")
        if la == lb == 0:
            return {"winner": "tie", "reason": "both empty"}
        if la >= lb:
            return {"winner": "a", "reason": "len(a) >= len(b)"}
        return {"winner": "b", "reason": "len(b) > len(a)"}


# =============================================================================
# Config
# =============================================================================
APP_VERSION = os.getenv("APP_VERSION", "2025-09-18")

# Resiliência (timeouts/retries)
PROVIDER_TIMEOUT_S: float = float(os.getenv("PROVIDER_TIMEOUT_S", "10"))
PROVIDER_MAX_RETRIES: int = int(os.getenv("PROVIDER_MAX_RETRIES", "0"))

# =============================================================================
# Circuit Breaker simples (local ao arquivo para não quebrar nada)
# =============================================================================
class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, reset_seconds: float = 30.0):
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_seconds
        self.failures = 0
        self.state: str = "closed"  # closed | half_open | open
        self._opened_at: Optional[float] = None

    def allow_request(self) -> bool:
        if self.state == "open":
            if self._opened_at is None:
                return False
            if (time.monotonic() - self._opened_at) >= self.reset_seconds:
                # janela de teste
                self.state = "half_open"
                return True
            return False
        return True  # closed ou half_open

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"
        self._opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.state in ("closed", "half_open") and self.failures >= self.fail_threshold:
            self.state = "open"
            self._opened_at = time.monotonic()

def compute_backoff(attempt: int, base: float = 0.2, factor: float = 2.0, jitter: float = 0.1) -> float:
    """Exponential backoff com jitter. attempt começa em 1."""
    expo = base * (factor ** max(0, attempt - 1))
    jitter_val = random.uniform(-jitter, jitter) * expo
    return max(0.0, expo + jitter_val)

# Um breaker por provider
_CB: Dict[str, CircuitBreaker] = {
    "openai": CircuitBreaker(),
    "gemini": CircuitBreaker(),
    "echo": CircuitBreaker(),
}

# =============================================================================
# App
# =============================================================================
app = FastAPI(title="Integração_Gem_GPT", version=APP_VERSION)

# =============================================================================
# Middleware: X-Request-ID
# =============================================================================
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or request.headers.get("x-request-id") or uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

# =============================================================================
# Métricas (Prometheus-like)
# =============================================================================
_METRICS: Dict[str, int] = {
    "ask_requests_success_echo": 0,
    "ask_requests_success_openai": 0,
    "ask_requests_success_gemini": 0,
    "ask_requests_error_echo": 0,
    "ask_requests_error_openai": 0,
    "ask_requests_error_gemini": 0,
    "ask_provider_error_openai": 0,
    "ask_provider_error_gemini": 0,
    "ask_provider_error_echo": 0,
    "ask_circuit_opens_openai": 0,
    "ask_circuit_opens_gemini": 0,
    "ask_retries_total": 0,
}

def _inc(key: str) -> None:
    _METRICS[key] = _METRICS.get(key, 0) + 1

def _metrics_record(provider: str, ok: bool) -> None:
    key = f'ask_requests_{"success" if ok else "error"}_{provider}'
    _inc(key)

# =============================================================================
# Helpers
# =============================================================================
def _to_text(maybe: Any) -> str:
    """
    Converte respostas de clients (string OU dict) em texto.
    Suporta:
      - OpenAI chat/completions: dict["choices"][0]["message"]["content"]
      - OpenAI legacy: dict["choices"][0]["text"]
      - Gemini: dict["candidates"][0]["content"]["parts"][0]["text"]
      - Campos comuns: "answer", "text", "content"
    """
    if maybe is None:
        return ""
    if isinstance(maybe, str):
        return maybe

    if isinstance(maybe, dict):
        # campos "answer"/"text"/"content"
        for k in ("answer", "text", "content"):
            v = maybe.get(k)
            if isinstance(v, str) and v.strip():
                return v

        # OpenAI Chat
        try:
            choices = maybe.get("choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0] or {}
                msg = ch0.get("message") or {}
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
                t = ch0.get("text")
                if isinstance(t, str) and t.strip():
                    return t
        except Exception:
            pass

        # Gemini
        try:
            cands = maybe.get("candidates")
            if isinstance(cands, list) and cands:
                content = (cands[0] or {}).get("content") or {}
                parts = content.get("parts")
                if isinstance(parts, list) and parts:
                    t = parts[0].get("text")
                    if isinstance(t, str) and t.strip():
                        return t
        except Exception:
            pass

        # fallback
        return str(maybe)

    return str(maybe)

# Compat: testes chamam app.main.openai_configured / gemini_configured
def openai_configured() -> bool:
    try:
        return bool(openai_is_configured())
    except Exception:
        return bool(os.getenv("OPENAI_API_KEY"))

def gemini_configured() -> bool:
    try:
        return bool(gemini_is_configured())
    except Exception:
        return bool(os.getenv("GEMINI_API_KEY"))

# =============================================================================
# Provider wrapper com CB/timeout/retry (sem passar kwargs como 'model')
# =============================================================================
async def _provider_call(name: str, prompt: str) -> Dict[str, Any]:
    cb = _CB.get(name)
    if cb and not cb.allow_request():
        _inc(f"ask_provider_error_{name}")
        raise RuntimeError(f"{name} circuit_open")

    async def _call_once() -> Any:
        if name == "openai":
            return await ask_openai(prompt)
        if name == "gemini":
            return await ask_gemini(prompt)
        if name == "echo":
            await asyncio.sleep(0.001)
            return prompt
        raise ValueError(f"provider desconhecido: {name}")

    last_exc: Optional[Exception] = None

    for attempt in range(1, PROVIDER_MAX_RETRIES + 2):  # 1 try + N retries
        try:
            raw = await asyncio.wait_for(_call_once(), timeout=PROVIDER_TIMEOUT_S)
            txt = _to_text(raw).strip()
            _metrics_record(name, True)
            if cb:
                cb.record_success()
            return {"provider": name, "answer": txt}
        except asyncio.TimeoutError:
            last_exc = TimeoutError(f"{name} timeout after {PROVIDER_TIMEOUT_S}s")
        except Exception as e:
            last_exc = e

        # falha -> contabiliza no CB
        if cb:
            prev_state = cb.state
            cb.record_failure()
            if cb.state == "open" and prev_state != "open":
                _inc(f"ask_circuit_opens_{name}")

        # decide retry
        if attempt <= PROVIDER_MAX_RETRIES:
            _inc("ask_retries_total")
            await asyncio.sleep(compute_backoff(attempt))
            continue

        # esgotou
        _inc(f"ask_provider_error_{name}")
        _metrics_record(name, False)
        if isinstance(last_exc, TimeoutError):
            raise last_exc
        raise RuntimeError(f"{name} error: {str(last_exc)}") from last_exc


# =============================================================================
# Rotas básicas
# =============================================================================
@app.get("/")
def root():
    return {"status": "live"}

@app.get("/ready")
def ready(request: Request, response: Response):
    rid = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
    if rid:
        response.headers["x-request-id"] = rid
    return {"status": "ready"}

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "ts": int(time.time()),
        "providers": {
            "openai_configured": openai_configured(),
            "gemini_configured": gemini_configured(),
        },
        "metrics": _METRICS,
    }

@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """
    Formato Prometheus:
      ask_requests_total{provider="echo",status="success"} <n>
      ask_provider_errors{provider="openai"} <n>
      ask_circuit_open{provider="openai"} 0|1
      ask_circuit_opens_total{provider="openai"} <n>
      ask_retries_total <n>
    """
    lines = []

    # pedidos por provider/status
    lines.append('# HELP ask_requests_total Número de requisições /ask por provider e status')
    lines.append('# TYPE ask_requests_total counter')
    for prov in ("echo", "openai", "gemini"):
        for status in ("success", "error"):
            key = f"ask_requests_{status}_{prov}"
            val = _METRICS.get(key, 0)
            lines.append(f'ask_requests_total{{provider="{prov}",status="{status}"}} {val}')

    # erros por provider
    lines.append('# HELP ask_provider_errors Número de erros por provider')
    lines.append('# TYPE ask_provider_errors counter')
    for prov in ("openai", "gemini", "echo"):
        key = f"ask_provider_error_{prov}"
        val = _METRICS.get(key, 0)
        lines.append(f'ask_provider_errors{{provider="{prov}"}} {val}')

    # estado atual do CB (gauge 0/1)
    lines.append('# HELP ask_circuit_open Estado do circuito por provider (0=closed/half_open, 1=open)')
    lines.append('# TYPE ask_circuit_open gauge')
    for prov in ("openai", "gemini"):
        state = _CB[prov].state
        lines.append(f'ask_circuit_open{{provider="{prov}"}} {1 if state=="open" else 0}')

    # total de aberturas
    lines.append('# HELP ask_circuit_opens_total Número de aberturas de circuito')
    lines.append('# TYPE ask_circuit_opens_total counter')
    for prov in ("openai", "gemini"):
        key = f"ask_circuit_opens_{prov}"
        val = _METRICS.get(key, 0)
        lines.append(f'ask_circuit_opens_total{{provider="{prov}"}} {val}')

    # retries globais
    lines.append('# HELP ask_retries_total Número de tentativas extras (retries)')
    lines.append('# TYPE ask_retries_total counter')
    lines.append(f'ask_retries_total {_METRICS.get("ask_retries_total", 0)}')

    return PlainTextResponse("\n".join(lines) + "\n")


# =============================================================================
# /ask
# =============================================================================
class AskPayload(BaseModel):
    prompt: str

@app.post("/ask")
async def ask_post(payload: AskPayload, provider: str = "auto"):
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")

    have_openai = openai_configured()
    have_gemini = gemini_configured()
    provider = (provider or "auto").lower()

    # ---- Provider explícito ----
    if provider == "openai":
        if not have_openai:
            _inc("ask_provider_error_openai")
            _metrics_record("openai", False)
            return JSONResponse(status_code=503, content={"detail": "openai_api_key não configurada"})
        try:
            r = await _provider_call("openai", prompt)
            return {"provider": r["provider"], "answer": r["answer"]}
        except Exception as e:
            return JSONResponse(status_code=502, content={"detail": str(e)})

    if provider == "gemini":
        if not have_gemini:
            _inc("ask_provider_error_gemini")
            _metrics_record("gemini", False)
            return JSONResponse(status_code=503, content={"detail": "gemini_api_key não configurada"})
        try:
            r = await _provider_call("gemini", prompt)
            return {"provider": r["provider"], "answer": r["answer"]}
        except Exception as e:
            return JSONResponse(status_code=502, content={"detail": str(e)})

    if provider == "echo":
        try:
            r = await _provider_call("echo", prompt)
            return {"provider": r["provider"], "answer": r["answer"]}
        except Exception as e:
            return JSONResponse(status_code=502, content={"detail": str(e)})

    # ---- Provider auto (fallback) ----
    if provider == "auto":
        # ambos off => 503 com detalhe do GEMINI (conforme testes esperavam)
        if not have_openai and not have_gemini:
            _inc("ask_provider_error_openai"); _metrics_record("openai", False)
            _inc("ask_provider_error_gemini"); _metrics_record("gemini", False)
            return JSONResponse(status_code=503, content={"detail": "gemini_api_key não configurada"})

        # tenta openai -> gemini
        if have_openai:
            try:
                r = await _provider_call("openai", prompt)
                return {"provider": r["provider"], "answer": r["answer"]}
            except Exception:
                pass
        if have_gemini:
            try:
                r = await _provider_call("gemini", prompt)
                return {"provider": r["provider"], "answer": r["answer"]}
            except Exception as e:
                return JSONResponse(status_code=503, content={"detail": str(e)})

        return JSONResponse(status_code=503, content={"detail": "Nenhum provider disponível."})

    # inválido
    return JSONResponse(status_code=400, content={"detail": "Parâmetros inválidos: provider=auto|openai|gemini|echo"})


# =============================================================================
# /duel
# =============================================================================
class DuelPayload(BaseModel):
    prompt: str

@app.post("/duel")
async def duel_post(payload: DuelPayload):
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")

    a_on = openai_configured()
    g_on = gemini_configured()

    # Nenhum provider disponível -> 502 com corpo detalhado
    if not a_on and not g_on:
        detail = {
            "mode": "duel",
            "responses": {
                "openai": {"ok": False, "answer": None},
                "gemini": {"ok": False, "answer": None},
            },
            "verdict": {"winner": "none", "reason": "no providers"},
        }
        return JSONResponse(status_code=502, content={"detail": detail})

    # Executa provedores disponíveis (usa _provider_call, que os testes monkeypatcham)
    a_ans, g_ans = None, None

    if a_on:
        try:
            r = await _provider_call("openai", prompt)
            a_ans = r.get("answer") or ""
        except Exception:
            a_ans = None

    if g_on:
        try:
            r = await _provider_call("gemini", prompt)
            g_ans = r.get("answer") or ""
        except Exception:
            g_ans = None

    # Chama judge_answers (pode ser async/sync; testes monkeypatcham)
    if inspect.iscoroutinefunction(judge_answers):
        ver = await judge_answers(prompt, a_ans or "", g_ans or "")  # type: ignore
    else:
        ver = judge_answers(prompt, a_ans or "", g_ans or "")  # type: ignore

    raw_winner = (ver or {}).get("winner")
    # normaliza "a"/"b" -> "openai"/"gemini"
    winner_map = {
        "a": "openai", "A": "openai",
        "b": "gemini", "B": "gemini",
        "openai": "openai", "gemini": "gemini",
        "tie": "tie", "none": "none",
    }
    norm_winner = winner_map.get(str(raw_winner), "tie")
    reason = (ver or {}).get("reason", "")

    body = {
        "mode": "duel",
        "responses": {
            "openai": {"ok": bool(a_ans), "answer": a_ans},
            "gemini": {"ok": bool(g_ans), "answer": g_ans},
        },
        "winner": norm_winner,
        "reason": reason,
        "verdict": {"winner": norm_winner, "reason": reason},
    }
    return JSONResponse(body)

# app/main.py
from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
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

app = FastAPI(title="Integração_Gem_GPT", version=APP_VERSION)

# =============================================================================
# Middleware: X-Request-ID
# =============================================================================
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # métrica por rota (usa path literal, estável para nossos testes)
        route = request.url.path
        _http_metrics_record(route, elapsed_ms)
        # log estruturado (se ligado)
        if OBS_JSON_LOGS:
            try:
                log_event = {
                    "ts": int(time.time()),
                    "level": "INFO",
                    "msg": "http_request",
                    "method": request.method,
                    "path": route,
                    "status": getattr(response, "status_code", None),
                    "latency_ms": round(elapsed_ms, 2),
                    "request_id": req_id,
                }
                logger.info(json.dumps(log_event, ensure_ascii=False))
            except Exception:
                pass
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
    "ask_provider_timeout_openai": 0,
    "ask_provider_timeout_gemini": 0,
    "ask_provider_timeout_echo": 0,

}

def _inc(key: str) -> None:
    _METRICS[key] = _METRICS.get(key, 0) + 1

def _metrics_record(provider: str, ok: bool) -> None:
    key = f'ask_requests_{"success" if ok else "error"}_{provider}'
    _inc(key)

# Métricas HTTP (por rota)
_HTTP_METRICS = {
    "http_requests_total": {},               # route -> count
    "http_request_latency_ms_sum": {},       # route -> sum
    "http_request_latency_ms_count": {},     # route -> count
}

def _http_metrics_record(route: str, latency_ms: float):
    r = route or "unknown"
    _HTTP_METRICS["http_requests_total"][r] = _HTTP_METRICS["http_requests_total"].get(r, 0) + 1
    _HTTP_METRICS["http_request_latency_ms_sum"][r] = _HTTP_METRICS["http_request_latency_ms_sum"].get(r, 0.0) + float(latency_ms)
    _HTTP_METRICS["http_request_latency_ms_count"][r] = _HTTP_METRICS["http_request_latency_ms_count"].get(r, 0) + 1

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

async def _provider_call(name: str, prompt: str) -> Dict[str, Any]:
    """
    Wrapper compatível com monkeypatch dos testes. NÃO passa kwargs como 'model',
    para evitar TypeError quando os testes substituem ask_*.
    Também atualiza métricas de sucesso/erro e provider_error_*.
    """
    try:
        if name == "openai":
            raw = await ask_openai(prompt)
        elif name == "gemini":
            raw = await ask_gemini(prompt)
        elif name == "echo":
            await asyncio.sleep(0.001)
            raw = prompt
        else:
            raise ValueError(f"provider desconhecido: {name}")

        txt = _to_text(raw).strip()
        _metrics_record(name, True)
        return {"provider": name, "answer": txt}
    except Exception as e:
        _inc(f"ask_provider_error_{name}")
        _metrics_record(name, False)
        # Para /ask?provider=gemini no teste de "Rate limit", o detail precisa conter o texto.
        raise RuntimeError(str(e)) from e

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
    """
    lines = []
    lines.append('# HELP ask_requests_total Número de requisições /ask por provider e status')
    lines.append('# TYPE ask_requests_total counter')
    for prov in ("echo", "openai", "gemini"):
        for status in ("success", "error"):
            key = f"ask_requests_{status}_{prov}"
            val = _METRICS.get(key, 0)
            lines.append(f'ask_requests_total{{provider="{prov}",status="{status}"}} {val}')
    lines.append('# HELP ask_provider_errors Número de erros por provider')
    lines.append('# TYPE ask_provider_errors counter')
    for prov in ("openai", "gemini", "echo"):
        key = f"ask_provider_error_{prov}"
        val = _METRICS.get(key, 0)
        lines.append(f'ask_provider_errors{{provider="{prov}"}} {val}')
    return PlainTextResponse("\n".join(lines) + "\n")

        # ---- Sprint 2: HTTP metrics ----
    lines.append('# HELP http_requests_total Número de requisições HTTP por rota')
    lines.append('# TYPE http_requests_total counter')
    for route, val in _HTTP_METRICS["http_requests_total"].items():
        lines.append(f'http_requests_total{{route="{route}"}} {val}')

    lines.append('# HELP http_request_latency_ms_sum Soma das latências (ms) por rota')
    lines.append('# TYPE http_request_latency_ms_sum counter')
    for route, val in _HTTP_METRICS["http_request_latency_ms_sum"].items():
        lines.append(f'http_request_latency_ms_sum{{route="{route}"}} {val}')

    lines.append('# HELP http_request_latency_ms_count Contagem de amostras de latência por rota')
    lines.append('# TYPE http_request_latency_ms_count counter')
    for route, val in _HTTP_METRICS["http_request_latency_ms_count"].items():
        lines.append(f'http_request_latency_ms_count{{route="{route}"}} {val}')

    lines.append('# HELP ask_provider_timeouts Número de timeouts por provider')
    lines.append('# TYPE ask_provider_timeouts counter')
    for prov in ("openai", "gemini", "echo"):
        key = f"ask_provider_timeout_{prov}"
        val = _METRICS.get(key, 0)
        lines.append(f'ask_provider_timeouts{{provider="{prov}"}} {val}')


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
            # IMPORTANTE: incrementar ambos (provider_error e request_error)
            _inc("ask_provider_error_openai")
            _metrics_record("openai", False)
            return JSONResponse(status_code=503, content={"detail": "openai_api_key não configurada"})
        try:
            r = await _provider_call("openai", prompt)
            return {"provider": r["provider"], "answer": r["answer"]}
        except Exception as e:
            # erro real do provider explicitamente selecionado -> 502
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
            _inc("ask_provider_error_openai")
            _metrics_record("openai", False)
            _inc("ask_provider_error_gemini")
            _metrics_record("gemini", False)
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


    # --- Sprint 2: Observability ---
import logging, json

def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() not in {"0", "false", "no", "off"}

OBS_JSON_LOGS = _env_bool("OBS_JSON_LOGS", True)

if OBS_JSON_LOGS:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orquestrador")


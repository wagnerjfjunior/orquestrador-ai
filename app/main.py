# app/main.py
from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

# Clientes locais
from .openai_client import ask as ask_openai, is_configured as openai_is_configured
from .gemini_client import ask as ask_gemini, is_configured as gemini_is_configured
from .judge import choose_winner_len, collab_fuse, contribution_ratio

APP_VERSION = os.getenv("APP_VERSION", "2025-09-18")
DEFAULT_MODEL_OPENAI = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MODEL_GEMINI = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
PROVIDER_TIMEOUT_S = float(os.getenv("PROVIDER_TIMEOUT_S", "22"))

app = FastAPI(title="Integração_Gem_GPT", version=APP_VERSION)

# --------------------------- Middleware: X-Request-ID ---------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or request.headers.get("x-request-id") or uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

# --------------------------- Métricas ---------------------------
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
}
def _inc(key: str) -> None:
    _METRICS[key] = _METRICS.get(key, 0) + 1

def _metrics_record(provider: str, ok: bool):
    key = f'ask_requests_{"success" if ok else "error"}_{provider}'
    _inc(key)

# --------------------------- Helpers ---------------------------
def _to_text(maybe: Any) -> str:
    """
    Converte respostas de clients (string OU dict) em texto.
    Suporta:
      - OpenAI chat/completions: dict["choices"][0]["message"]["content"]
      - OpenAI responses: dict["choices"][0]["text"] (modelos antigos)
      - Gemini: dict["candidates"][0]["content"]["parts"][0]["text"]
      - Campos comuns: "answer", "text", "content"
    """
    if maybe is None:
        return ""

    if isinstance(maybe, str):
        return maybe

    if isinstance(maybe, dict):
        # 1) Alguns clients já devolvem {"answer": "..."}
        for k in ("answer", "text", "content"):
            v = maybe.get(k)
            if isinstance(v, str) and v.strip():
                return v

        # 2) OpenAI Chat Completions
        try:
            choices = maybe.get("choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0] or {}
                # a) formato chat moderno
                msg = ch0.get("message") or {}
                c = msg.get("content")
                if isinstance(c, str) and c.strip():
                    return c
                # b) alguns retornam 'text' diretamente no choice
                t = ch0.get("text")
                if isinstance(t, str) and t.strip():
                    return t
        except Exception:
            pass

        # 3) Gemini: candidates[0].content.parts[0].text
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

        # 4) fallback: stringifica o dict (evita crash e dá visibilidade)
        return str(maybe)

    # 5) qualquer outro tipo → string
    return str(maybe)

async def _provider_call(name: str, prompt: str) -> Dict[str, Any]:
    async def _call() -> Any:  # pode retornar str OU dict
        if name == "openai":
            return await ask_openai(prompt, model=DEFAULT_MODEL_OPENAI)
        if name == "gemini":
            return await ask_gemini(prompt, model=DEFAULT_MODEL_GEMINI)
        if name == "echo":
            await asyncio.sleep(0.001)
            return prompt
        raise ValueError(f"provider desconhecido: {name}")

    try:
        raw = await asyncio.wait_for(_call(), timeout=PROVIDER_TIMEOUT_S)
        txt = _to_text(raw)
        _metrics_record(name, True)
        return {"provider": name, "answer": txt.strip()}
    except asyncio.TimeoutError as e:
        _inc(f"ask_provider_error_{name}")
        _metrics_record(name, False)
        raise TimeoutError(f"{name} timeout after {PROVIDER_TIMEOUT_S}s") from e
    except Exception as e:
        _inc(f"ask_provider_error_{name}")
        _metrics_record(name, False)
        raise RuntimeError(f"{name} error: {str(e)}") from e

async def _race_two(prompt: str, have_openai: bool, have_gemini: bool):
    ans_o = None
    ans_g = None
    errors = {}
    tasks = []
    if have_openai:
        tasks.append(asyncio.create_task(_provider_call("openai", prompt)))
    if have_gemini:
        tasks.append(asyncio.create_task(_provider_call("gemini", prompt)))
    if not tasks:
        raise HTTPException(status_code=503, detail="Nenhum provider configurado (OPENAI_API_KEY/GEMINI_API_KEY).")

    for t in asyncio.as_completed(tasks):
        try:
            r = await t
            if r["provider"] == "openai":
                ans_o = r["answer"]
            else:
                ans_g = r["answer"]
        except Exception as e:
            msg = str(e)
            # tenta mapear o provider pelo texto
            ml = msg.lower()
            if "openai" in ml and "openai" not in errors:
                errors["openai"] = msg
            elif "gemini" in ml and "gemini" not in errors:
                errors["gemini"] = msg
            else:
                # fallback heurístico
                if "openai" not in errors and have_openai and ans_o is None:
                    errors["openai"] = msg
                elif "gemini" not in errors and have_gemini and ans_g is None:
                    errors["gemini"] = msg
    return ans_o, ans_g, errors


def _extract_prompt(request: Request, body: Optional[Dict[str, Any]]) -> str:
    if request.headers.get("content-type", "").startswith("application/json"):
        prompt = (body or {}).get("prompt", "")
    else:
        prompt = request.query_params.get("prompt", "")
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")
    return prompt

# --------------------------- Rotas ---------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "ts": int(time.time()),
        "providers": {
            "openai_configured": openai_is_configured(),
            "gemini_configured": gemini_is_configured(),
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

@app.post("/ask")
async def ask_post(request: Request) -> JSONResponse:
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    prompt = _extract_prompt(request, body)

    provider = (request.query_params.get("provider", "") or "auto").lower()
    strategy = (request.query_params.get("strategy", "") or "").lower()

    have_openai = openai_is_configured()
    have_gemini = gemini_is_configured()

    # ---- Estratégias ----
    if strategy in {"heuristic", "crossvote", "collab"}:
        
        ans_o, ans_g, errs = await _race_two(prompt, have_openai, have_gemini)
        if not ans_o and not ans_g:
            raise HTTPException(status_code=502, detail={"message": "Falha nos providers", "errors": errs})


        if strategy in {"heuristic", "crossvote"}:
            winner = choose_winner_len(ans_o or "", ans_g or "")
            final = ans_o if winner == "openai" else ans_g
            return JSONResponse({
                "strategy_used": strategy,
                "prompt": prompt,
                "responses": {"openai": ans_o, "gemini": ans_g},
                "verdict": {"winner": winner, "reason": "length"},
                "final": final,
            })

        # collab
        fused = collab_fuse({"openai": ans_o or "", "gemini": ans_g or ""})
        ratios = contribution_ratio(fused, {"openai": ans_o or "", "gemini": ans_g or ""})
        return JSONResponse({
            "strategy_used": "collab",
            "prompt": prompt,
            "responses": {"openai": ans_o, "gemini": ans_g},
            "contribution": ratios,
            "final": fused,
        })

    # ---- Provider explícito ----
    if provider in {"openai", "gemini", "echo"}:
        try:
            r = await _provider_call(provider, prompt)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        return JSONResponse({"provider": r["provider"], "answer": r["answer"]})

    # ---- Provider auto (compat) ----
    if provider == "auto":
        if have_openai:
            try:
                r = await _provider_call("openai", prompt)
                return JSONResponse({"provider": r["provider"], "answer": r["answer"]})
            except Exception:
                pass
        if have_gemini:
            r = await _provider_call("gemini", prompt)
            return JSONResponse({"provider": r["provider"], "answer": r["answer"]})
        raise HTTPException(status_code=503, detail="Nenhum provider disponível.")

    raise HTTPException(status_code=400, detail="Parâmetros inválidos: use provider=auto|openai|gemini|echo ou strategy=heuristic|crossvote|collab")

@app.get("/ask")
async def ask_get(request: Request) -> JSONResponse:
    # delega para POST mantendo query params; lê prompt da query
    body = {"prompt": request.query_params.get("prompt", "")}
    # constrói um Request “equivalente” para reaproveitar a função
    return await ask_post(Request(scope=request.scope, receive=request.receive, send=None))

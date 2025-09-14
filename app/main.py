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


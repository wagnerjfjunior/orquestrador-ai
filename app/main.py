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

# app/main.py
from typing import Dict, Any, List
import time
from fastapi import FastAPI, Body, HTTPException
from app.observability import TraceMiddleware, logger
from app.metrics import setup_metrics, record_ask
from app.config import settings
from app.openai_client import ask_openai, is_configured as openai_configured
from app.gemini_client import ask_gemini, is_configured as gemini_configured

app = FastAPI(
    title="orquestrador-ai",
    version=settings.APP_VERSION,
    description="Orquestrador multi-IA com observabilidade e métricas",
)

# --- Middlewares e métricas ---
app.add_middleware(TraceMiddleware)
setup_metrics(app)

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


def _provider_is_configured(name: str) -> bool:
    name = name.lower()
    if name == "openai":
        return openai_configured()
    if name == "gemini":
        return gemini_configured()
    if name == "echo":
        return True
    return False


def _provider_call(name: str, prompt: str) -> Dict[str, Any]:
    name = name.lower()

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


@app.post("/ask", tags=["ask"])
def ask(provider: str = "echo", payload: dict = Body(...), use_fallback: bool = True):
    """
    - provider: echo | openai | gemini | auto
    - use_fallback: só tem efeito quando provider=auto (explícito ignora fallback)
    """
    prompt = payload.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório no corpo JSON.")

    chain = _fallback_chain(provider)
    is_auto = (provider or "").lower() == "auto"

    start = time.perf_counter()
    last_error: Exception | None = None

    # Provider explícito: sem fallback, registra métricas direto
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

    # provider=auto: tenta cadeia com fallback
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

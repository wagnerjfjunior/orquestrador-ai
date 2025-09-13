# app/main.py
from typing import Callable, Dict, Any, List
from fastapi import FastAPI, Body, HTTPException
from app.observability import TraceMiddleware, logger
from app.metrics import setup_metrics
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
    """Usado no teste: deve retornar {'status': 'live'}"""
    logger.info("root.live")
    return {"status": "live"}

@app.get("/health", tags=["infra"])
def health():
    """Health check simples"""
    logger.info("health.ok")
    return {"status": "ok"}

@app.get("/ready", tags=["infra"])
def readiness():
    """Readiness para probes"""
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
    """
    Chama o provedor indicado e retorna resposta normalizada.
    Lança RuntimeError em erro do provedor.
    Lança HTTPException 503 se não configurado.
    """
    name = name.lower()

    if name == "echo":
        logger.info("ask.echo", prompt=prompt)
        return {"provider": "echo", "answer": prompt, "output": prompt}

    if name == "openai":
        if not openai_configured():
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY não configurada.")
        return ask_openai(prompt)

    if name == "gemini":
        if not gemini_configured():
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY não configurada.")
        return ask_gemini(prompt)

    raise HTTPException(status_code=400, detail=f"Provider não suportado: {name}")


def _fallback_chain(primary: str | None) -> List[str]:
    """
    Constrói a cadeia de fallback.
    - Se provider='auto' ou None: usa settings.PROVIDER_FALLBACK como está.
    - Se provider específico: tenta ele primeiro e depois completa com os da lista,
      sem duplicar e mantendo ordem.
    """
    order = [p.lower() for p in settings.PROVIDER_FALLBACK]
    if not primary or primary.lower() == "auto":
        return order
    seq = [primary.lower()] + [p for p in order if p != primary.lower()]
    return seq


# --- Orquestrador: endpoint /ask ---
@app.post("/ask", tags=["ask"])
def ask(provider: str = "echo", payload: dict = Body(...), use_fallback: bool = True):
    """
    Endpoint principal do orquestrador.

    Parâmetros de query:
      - provider: echo | openai | gemini | auto
      - use_fallback: se True, tenta próximos provedores da cadeia se houver erro do atual.

    Exemplos:
      - /ask?provider=echo
      - /ask?provider=openai
      - /ask?provider=gemini
      - /ask?provider=auto  (segue settings.PROVIDER_FALLBACK)
    """
    prompt = payload.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório no corpo JSON.")

    chain = _fallback_chain(provider)

    last_error: Exception | None = None
    for p in chain:
        logger.info("ask.try_provider", provider=p)
        # Se não for para usar fallback e p não for o primeiro da cadeia, pare.
        if not use_fallback and p != chain[0]:
            break

        # Se o provedor não está configurado, pule para o próximo (mas registre).
        if not _provider_is_configured(p):
            logger.info("ask.provider_not_configured", provider=p)
            last_error = HTTPException(status_code=503, detail=f"Provider não configurado: {p}")
            continue

        try:
            resp = _provider_call(p, prompt)
            logger.info("ask.provider_success", provider=p)
            return resp
        except HTTPException as http_exc:
            # 4xx/5xx “nossos” (não do provedor) — apenas propaga se for o primeiro ou fallback desativado
            logger.info("ask.http_exception", provider=p, status=http_exc.status_code)
            last_error = http_exc
            if not use_fallback:
                raise http_exc
            # senão tenta o próximo da cadeia
        except RuntimeError as runtime_err:
            # Erros de provedor são normalizados como RuntimeError nos clients
            logger.info("ask.provider_runtime_error", provider=p, error=str(runtime_err))
            last_error = runtime_err
            if not use_fallback:
                raise HTTPException(status_code=502, detail=str(runtime_err))

    # Se chegou aqui, todos falharam
    if isinstance(last_error, HTTPException):
        # se o último foi HTTPException (ex.: não configurado), propaga
        raise last_error
    # caso contrário, erro de provedor → 502
    detail = str(last_error) if last_error else "Falha ao atender requisição em todos os provedores."
    raise HTTPException(status_code=502, detail=detail)

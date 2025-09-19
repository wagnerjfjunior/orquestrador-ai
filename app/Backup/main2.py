# app/main.py
from __future__ import annotations

import os
import time
import uuid
import asyncio
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

# ============================
# Observabilidade / Métricas
# ============================

ASK_TOTAL = 0
ASK_PROVIDER_ERRORS = {
    "openai": 0,
    "gemini": 0,
    "echo": 0,
}

def _inc(metric: str, provider: Optional[str] = None) -> None:
    global ASK_TOTAL, ASK_PROVIDER_ERRORS
    if metric == "ask_total":
        ASK_TOTAL += 1
    elif metric == "ask_provider_error" and provider:
        ASK_PROVIDER_ERRORS[provider] = ASK_PROVIDER_ERRORS.get(provider, 0) + 1


# ============================
# App e Middleware de Request-ID
# ============================

app = FastAPI(title="Orquestrador AI")

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    # Propaga X-Request-ID ou gera um
    req_id = request.headers.get("X-Request-ID") or request.headers.get("x-request-id")
    if not req_id:
        req_id = f"req-{uuid.uuid4()}"
    request.state.request_id = req_id

    # Executa a request
    response: Response
    try:
        response = await call_next(request)
    except Exception:
        # Em caso de erro, ainda devolvemos o header
        response = JSONResponse({"detail": "internal error"}, status_code=500)

    # Garante o header na resposta
    response.headers["X-Request-ID"] = req_id
    return response


# ============================
# Helpers de configuração
# (usados nos testes com monkeypatch)
# ============================

def openai_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def gemini_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


# ============================
# Provedores (assíncronos)
# ============================

async def ask_openai(prompt: str) -> Dict[str, Any]:
    """
    Chamada real ao OpenAI (simplificada).
    Mantida assíncrona para ser compatível com testes que monkeypatcham corrotinas.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY não configurada")

    # Evita importar SDKs aqui — simula latência/chamada.
    await asyncio.sleep(0)  # yield para o loop
    # Resposta fake só para os testes; em produção, integre com seu cliente.
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "answer": f"[openai] {prompt}",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


async def ask_gemini(prompt: str) -> Dict[str, Any]:
    """
    Chamada real ao Gemini (simplificada).
    Mantida assíncrona para ser compatível com testes que monkeypatcham corrotinas.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY não configurada")

    await asyncio.sleep(0)
    return {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "answer": f"[gemini] {prompt}",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


async def ask_echo(prompt: str) -> Dict[str, Any]:
    """
    Provedor 'echo' para testes — nunca exige API key.
    """
    await asyncio.sleep(0)
    return {"provider": "echo", "answer": prompt}


# ============================
# Funções utilitárias
# ============================

async def _provider_call(name: str, prompt: str) -> Dict[str, Any]:
    """
    Wrapper para o call do provedor; os testes monkeypatcham esse símbolo.
    """
    if name == "openai":
        return await ask_openai(prompt)
    if name == "gemini":
        return await ask_gemini(prompt)
    if name == "echo":
        return await ask_echo(prompt)
    raise ValueError(f"provider desconhecido: {name}")


def _choose_winner_by_len(a: str, b: str) -> str:
    """
    Heurística besta só para testes (sem juiz).
    """
    return "openai" if len(a) >= len(b) else "gemini"


# ============================
# Endpoints básicos
# ============================

@app.get("/")
def root():
    return {"name": "orquestrador-ai", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready(request: Request):
    # Só para validar a propagação do header; retorna 200
    return {"ready": True, "request_id": request.state.request_id}


# ============================
# Métricas (formato simples tipo Prometheus)
# ============================

@app.get("/metrics")
def metrics():
    lines = [
        "# HELP ask_total Número total de requisições /ask",
        "# TYPE ask_total counter",
        f"ask_total {ASK_TOTAL}",
        "# HELP ask_provider_errors Número de erros por provider",
        "# TYPE ask_provider_errors counter",
    ]
    for prov, val in ASK_PROVIDER_ERRORS.items():
        lines.append(f'ask_provider_errors{{provider="{prov}"}} {val}')
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


# ============================
# Endpoint /ask
# ============================

@app.post("/ask")
async def ask(req: Request):
    """
    Query params:
      - provider=echo|openai|gemini|auto  (default: auto)
      - use_fallback=true|false           (default: true)
      - strategy=heuristic|crossvote|collab (compatível com sua execução via curl)
    JSON body:
      - { "prompt": "..." }
    """
    global ASK_TOTAL
    _inc("ask_total")

    params = dict(req.query_params)
    body = await req.json()
    prompt = body.get("prompt")
    provider = (params.get("provider") or "auto").lower()
    use_fallback = (params.get("use_fallback", "true").lower() != "false")
    strategy = (params.get("strategy") or "heuristic").lower()

    if not prompt:
        raise HTTPException(status_code=400, detail="Campo 'prompt' é obrigatório.")

    # Provider explícito "echo" ignora chaves e serve pros testes
    if provider == "echo":
        data = await _provider_call("echo", prompt)
        return JSONResponse(data)

    # Strategy "crossvote" ou "collab" — mantém compatibilidade com seus curls
    if strategy in ("crossvote", "collab"):
        # Para simplificar os testes, chamamos os provedores (se configurados)
        answers: Dict[str, str] = {}
        errors: Dict[str, str] = {}

        if openai_configured():
            try:
                r = await _provider_call("openai", prompt)
                answers["openai"] = r["answer"]
            except Exception as e:
                _inc("ask_provider_error", "openai")
                errors["openai"] = str(e)
        if gemini_configured():
            try:
                r = await _provider_call("gemini", prompt)
                answers["gemini"] = r["answer"]
            except Exception as e:
                _inc("ask_provider_error", "gemini")
                errors["gemini"] = str(e)

        if not answers:
            raise HTTPException(status_code=502, detail={"errors": errors or "no providers"})

        if strategy == "crossvote":
            # heurística simples de "voto": escolhe a mais longa
            o = answers.get("openai", "")
            g = answers.get("gemini", "")
            winner = _choose_winner_by_len(o, g)
            return JSONResponse({
                "strategy_used": "crossvote",
                "prompt": prompt,
                "final_responses": answers,
                "verdict": {"winner": winner, "reason": "Heurística de exemplo (comprimento)"}
            })

        # collab: funde respostas de forma simples (concat deduplicada)
        parts = []
        for k in ("openai", "gemini"):
            if k in answers and answers[k] not in parts:
                parts.append(answers[k])
        fused = "\n\n".join(parts)
        return JSONResponse({
            "strategy_used": "collab",
            "prompt": prompt,
            "final_answer": fused,
            "final_responses": answers,
            "verdict": {"winner": "openai" if "openai" in answers else "gemini", "reason": "heurística simples"}
        })

    # Fluxo "básico" / testes: provider=openai|gemini|auto
    async def try_openai() -> Optional[Dict[str, Any]]:
        if not openai_configured():
            return None
        try:
            return await _provider_call("openai", prompt)
        except Exception as e:
            _inc("ask_provider_error", "openai")
            if not use_fallback:
                # Se não quer fallback, propaga erro como 502
                raise HTTPException(status_code=502, detail=str(e))
            return None

    async def try_gemini() -> Optional[Dict[str, Any]]:
        if not gemini_configured():
            return None
        try:
            return await _provider_call("gemini", prompt)
        except Exception as e:
            _inc("ask_provider_error", "gemini")
            if not use_fallback:
                raise HTTPException(status_code=502, detail=str(e))
            return None

    if provider == "openai":
        r = await try_openai()
        if r is None:
            if use_fallback:
                # tenta fallback no gemini
                r = await try_gemini()
        if r is None:
            raise HTTPException(status_code=502, detail="Nenhum provedor disponível")
        return JSONResponse(r)

    if provider == "gemini":
        r = await try_gemini()
        if r is None:
            if use_fallback:
                r = await try_openai()
        if r is None:
            raise HTTPException(status_code=502, detail="Nenhum provedor disponível")
        return JSONResponse(r)

    # provider == auto: tenta openai, depois gemini
    r = await try_openai()
    if r is None:
        r = await try_gemini()
    if r is None:
        raise HTTPException(status_code=502, detail="Nenhum provedor disponível")
    return JSONResponse(r)


# ============================
# Endpoint /duel (testes)
# ============================

@app.post("/duel")
async def duel(req: Request):
    """
    Disputa entre openai e gemini e retorno do vencedor (heurística comprimento).
    Retorna 502 se nenhum provider estiver configurado.
    """
    body = await req.json()
    prompt = body.get("prompt") or ""

    have_openai = openai_configured()
    have_gemini = gemini_configured()

    if not have_openai and not have_gemini:
        raise HTTPException(status_code=502, detail="No providers configured")

    ans_a = ""
    ans_b = ""

    if have_openai:
        try:
            r = await _provider_call("openai", prompt)
            ans_a = r["answer"]
        except Exception:
            _inc("ask_provider_error", "openai")

    if have_gemini:
        try:
            r = await _provider_call("gemini", prompt)
            ans_b = r["answer"]
        except Exception:
            _inc("ask_provider_error", "gemini")

    winner = _choose_winner_by_len(ans_a, ans_b)
    return {"winner": "openai" if winner == "openai" else "gemini", "a_len": len(ans_a), "b_len": len(ans_b)}

# ==============================
# app/gemini_client.py
# Propósito:
# - Cliente Gemini (Google Generative AI)
# - Compatível com os testes: precisa retornar dict completo
#   {"provider":"gemini","model":<modelo>,"answer":<texto>,"usage":{}}
# - Expor: is_configured(), ask_gemini(), judge()
#
# Alterações nesta revisão:
# - ask_gemini retorna dict completo (não só {"answer": ...})
# - Mantido judge() síncrono como o judge.py espera
# ==============================
from __future__ import annotations

import os
import asyncio
from typing import Dict, Any, Optional

import google.generativeai as genai


# --------------------------------------------
# Config
# --------------------------------------------
_GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or None
_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if _GEMINI_API_KEY:
    genai.configure(api_key=_GEMINI_API_KEY)


def is_configured() -> bool:
    """Retorna True se a GEMINI_API_KEY estiver disponível."""
    return bool(_GEMINI_API_KEY)


# --------------------------------------------
# Helpers (sync) para chamar a API do Gemini
# --------------------------------------------
def _gemini_generate_sync(
    prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    mdl = genai.GenerativeModel(model)
    resp = mdl.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )
    text = getattr(resp, "text", None)
    if not text and hasattr(resp, "candidates") and resp.candidates:
        text = getattr(resp.candidates[0], "content", None)
        if hasattr(text, "parts") and text.parts:
            text = "".join(getattr(p, "text", "") for p in text.parts)
        elif text is None:
            text = ""
    return (text or "").strip()


def _gemini_generate_with_system_sync(
    system: str,
    user: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    composed = (
        f"[SYSTEM]\n{system.strip()}\n\n"
        f"[USER]\n{user.strip()}\n"
        "IMPORTANTE: Responda ESTRITAMENTE com um ÚNICO objeto JSON válido (RFC 8259) "
        "sem markdown e sem texto fora do JSON."
    )
    mdl = genai.GenerativeModel(model)
    resp = mdl.generate_content(
        composed,
        generation_config={"temperature": temperature},
    )
    text = getattr(resp, "text", None)
    return (text or "").strip()


# --------------------------------------------
# API Async usada pelo app
# --------------------------------------------
async def ask_gemini(
    prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: float = 25.0,
) -> Dict[str, Any]:
    """
    Gera uma resposta com o Gemini e retorna no formato esperado:
      { "provider":"gemini", "model":..., "answer":..., "usage":{} }
    """
    if not is_configured():
        raise RuntimeError("GEMINI_API_KEY não configurada")

    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_gemini_generate_sync, prompt, model=model, temperature=temperature),
            timeout=timeout,
        )
        return {
            "provider": "gemini",
            "model": model,
            "answer": text,
            "usage": {},
        }
    except asyncio.TimeoutError as te:
        raise RuntimeError("Timeout ao chamar Gemini.") from te
    except Exception:
        raise


# --------------------------------------------
# Função 'judge' usada pelo módulo judge.py
# --------------------------------------------
def judge(
    system: str,
    user: str,
    *,
    force_json: bool = True,
    temperature: float = 0.0,
    timeout: float = 20.0,
) -> str:
    if not is_configured():
        raise RuntimeError("GEMINI_API_KEY não configurada")
    return _gemini_generate_with_system_sync(
        system=system,
        user=user,
        model=_DEFAULT_MODEL,
        temperature=temperature,
    )
# Alias para compatibilidade com o orquestrador:
ask = ask_gemini

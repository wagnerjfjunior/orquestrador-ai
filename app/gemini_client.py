# =============================================================================
# File: app/gemini_client.py
# Version: 2025-09-14 16:25:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO CRÍTICA: Corrigido SyntaxError ('{' was never closed) no dicionário de retorno.
# - Refatorado para ser totalmente assíncrono.
# - Utiliza `await gmodel.generate_content_async` para a chamada de API.
# =============================================================================
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


async def ask_gemini(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,  # mantido para simetria
    **extra: Any,
) -> Dict[str, Any]:
    """
    Envia um prompt ao Gemini de forma assíncrona e retorna resposta normalizada.
    """
    mdl = model or settings.GEMINI_MODEL
    tmo = timeout or settings.PROVIDER_TIMEOUT

    logger.info("provider.gemini.request.async", model=mdl, timeout=tmo)

    try:
        gmodel = _build_model(mdl)

        generation_config = extra.get("generation_config")
        safety_settings = extra.get("safety_settings")

        resp = await gmodel.generate_content_async(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        content = (getattr(resp, "text", "") or "").strip()
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

        logger.info("provider.gemini.success.async", model=mdl)

        # DICIONÁRIO DE RETORNO CORRIGIDO
        return {
            "provider": "gemini",
            "model": mdl,
            "answer": content,
            "usage": usage,
        }
    except genai.types.generation_types.BlockedPromptException as e:
        logger.info("provider.gemini.blocked_prompt", error=str(e))
        raise RuntimeError("Prompt bloqueado pela política do Gemini.") from e
    except genai.types.generation_types.StopCandidateException as e:
        logger.info("provider.gemini.stop_candidate", error=str(e))
        raise RuntimeError("Geração interrompida pelo Gemini.") from e
    except Exception as e:
        logger.info("provider.gemini.unexpected_error", error=str(e))
        raise RuntimeError("Erro inesperado ao chamar o Gemini.") from e


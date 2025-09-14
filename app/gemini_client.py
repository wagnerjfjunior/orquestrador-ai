# app/gemini_client.py
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


def ask_gemini(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,  # mantido para simetria
    **extra: Any,
) -> Dict[str, Any]:
    """
    Envia um prompt ao Gemini e retorna resposta normalizada.

    Parâmetros:
      - prompt: texto do usuário
      - model: override do modelo (default: settings.GEMINI_MODEL)
      - timeout: mantido por simetria (a SDK atual não aceita timeout direto)
      - **extra: futuros parâmetros (ex.: generation_config)

    Retorno:
      {
        "provider": "gemini",
        "model": "<modelo>",
        "answer": "<texto>",
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
      }
    """
    mdl = model or settings.GEMINI_MODEL
    tmo = timeout or settings.PROVIDER_TIMEOUT

    logger.info("provider.gemini.request", model=mdl, timeout=tmo)

    try:
        gmodel = _build_model(mdl)

        # Permite passar configs adicionais (ex: generation_config={"temperature":0.3})
        generation_config = extra.get("generation_config")
        safety_settings = extra.get("safety_settings")

        resp = gmodel.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # A SDK expõe texto final em resp.text
        content = (getattr(resp, "text", "") or "").strip()

        # A SDK atual não retorna usage padronizado — deixamos None
        usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

        logger.info("provider.gemini.success", model=mdl)

        return {
            "provider": "gemini",
            "model": mdl,
            "answer": content,
            "usage": usage,
        }

    except genai.types.generation_types.BlockedPromptException as e:  # type: ignore[attr-defined]
        logger.info("provider.gemini.blocked_prompt", error=str(e))
        raise RuntimeError("Prompt bloqueado pela política do Gemini.") from e
    except genai.types.generation_types.StopCandidateException as e:  # type: ignore[attr-defined]
        logger.info("provider.gemini.stop_candidate", error=str(e))
        raise RuntimeError("Geração interrompida pelo Gemini.") from e
    except Exception as e:
        logger.info("provider.gemini.unexpected_error", error=str(e))
        raise RuntimeError("Erro inesperado ao chamar o Gemini.") from e

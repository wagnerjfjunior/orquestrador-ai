# =============================================================================
# File: app/openai_client.py
# Version: 2025-09-14 15:59:00 -03 (America/Sao_Paulo)
# Changes:
# - Refatorado para ser totalmente assíncrono.
# - Uso do AsyncOpenAI para chamadas não-bloqueantes.
# - Função _build_async_client para criar o cliente assíncrono.
# - ask_openai agora é uma função `async def`.
# - Utiliza `await client.chat.completions.create` para a chamada de API.
# =============================================================================
from __future__ import annotations

from typing import Any, Dict, Optional

from openai import APIConnectionError, APIStatusError, AuthenticationError, AsyncOpenAI, RateLimitError

from app.config import settings
from app.observability import logger


def is_configured() -> bool:
    """
    Retorna True se houver OPENAI_API_KEY configurada.
    """
    return bool(settings.OPENAI_API_KEY)


def _build_async_client(timeout: Optional[float] = None) -> AsyncOpenAI:
    """
    Constroi o cliente AsyncOpenAI com a API key do settings.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não configurada.")

    # Usa o cliente assíncrono para chamadas não-bloqueantes
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=timeout or settings.PROVIDER_TIMEOUT)


async def ask_openai(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Envia um prompt para o OpenAI Chat Completions de forma assíncrona.

    Parâmetros:
      - prompt: texto do usuário
      - model: override do modelo (default: settings.OPENAI_MODEL)
      - timeout: timeout em segundos para esta chamada (default: settings.PROVIDER_TIMEOUT)
      - **extra: espaço para parâmetros futuros (temperature, top_p, etc.)

    Retorno:
      {
        "provider": "openai",
        "model": "<modelo>",
        "answer": "<texto>",
        "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
      }

    Exceções:
      - RuntimeError em casos de erro de conexão, auth, rate limit, status != 2xx, ou outro erro inesperado.
    """
    mdl = model or settings.OPENAI_MODEL
    tmo = timeout or settings.PROVIDER_TIMEOUT

    client = _build_async_client(timeout=tmo)
    logger.info("provider.openai.request.async", model=mdl)

    try:
        # A biblioteca openai gerencia retries para o cliente async por padrão
        resp = await client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )

        content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

        logger.info(
            "provider.openai.success.async",
            model=mdl,
            total_tokens=usage["total_tokens"],
        )

        return {
            "provider": "openai",
            "model": mdl,
            "answer": content,
            "usage": usage,
        }

    except AuthenticationError as e:
        logger.info("provider.openai.auth_error", error=str(e))
        raise RuntimeError("Falha de autenticação na OpenAI (verifique OPENAI_API_KEY).") from e
    except RateLimitError as e:
        logger.info("provider.openai.rate_limit", error=str(e))
        raise RuntimeError("Rate limit atingido na OpenAI. Tente novamente mais tarde.") from e
    except APIStatusError as e:
        logger.info("provider.openai.api_status_error", status=e.status_code, error=str(e))
        raise RuntimeError(f"Erro de status na OpenAI: {e.status_code}.") from e
    except APIConnectionError as e:
        logger.info("provider.openai.connection_error", error=str(e))
        raise RuntimeError("Erro de conexão com a OpenAI.") from e
    except Exception as e:
        logger.info("provider.openai.unexpected_error", error=str(e))
        raise RuntimeError("Erro inesperado ao chamar a OpenAI.") from e


# app/openai_client.py
from __future__ import annotations

from typing import Optional, Dict, Any, Callable
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError, AuthenticationError
from app.config import settings
from app.observability import logger
from app.utils.retry import retry, RetryExceededError


def is_configured() -> bool:
    """
    Retorna True se houver OPENAI_API_KEY configurada.
    """
    return bool(settings.OPENAI_API_KEY)


def _build_client(timeout: Optional[float] = None) -> OpenAI:
    """
    Constroi o cliente OpenAI com a API key do settings.
    Permite sobrescrever timeout por chamada.
    """
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não configurada.")

    # openai-python v1.x
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    # Ajuste de timeout por requisição (create(..., timeout=...))
    # Mantemos simples aqui; se quiser timeout global, dá pra ajustar http_client no client.
    return client


def ask_openai(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Envia um prompt para o OpenAI Chat Completions e retorna resposta normalizada.

    Parâmetros:
      - prompt: texto do usuário
      - model: override do modelo (default: settings.OPENAI_MODEL)
      - timeout: timeout em segundos para esta chamada (default: settings.PROVIDER_TIMEOUT)
      - **extra: espaço para parâmetros futuros (temperature, top_p, etc.)
                aceita também _sleep (Callable[[float], None]) usado nos testes de retry

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

    # Permite testes injetarem um sleep no-op sem mudar a assinatura pública
    _sleep: Optional[Callable[[float], None]] = extra.pop("_sleep", None)

    client = _build_client(timeout=tmo)
    logger.info("provider.openai.request", model=mdl)

    def _do_call() -> Dict[str, Any]:
        # API Chat Completions (modelos como gpt-4o-mini)
        resp = client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            timeout=tmo,
            **extra,
        )

        content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

        logger.info(
            "provider.openai.success",
            model=mdl,
            total_tokens=usage["total_tokens"],
        )

        return {
            "provider": "openai",
            "model": mdl,
            "answer": content,
            "usage": usage,
        }

    try:
        # Retry leve apenas para erros transitórios de rede
        return retry(
            _do_call,
            retries=2,            # até 2 tentativas adicionais (total máx = 3)
            backoff_ms=200,       # exponencial simples: 200ms, depois 400ms
            retry_on=(APIConnectionError,),
            sleep=_sleep,         # nos testes, injetamos no-op para não atrasar
        )

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
        # Pode ser lançado antes do retry ou mesmo após esgotar tentativas em algum caminho
        logger.info("provider.openai.connection_error", error=str(e))
        raise RuntimeError("Erro de conexão com a OpenAI.") from e
    except RetryExceededError as e:
        logger.info("provider.openai.retry_exceeded", error=str(e))
        raise RuntimeError("Erro de conexão com a OpenAI (tentativas esgotadas).") from e
    except Exception as e:
        logger.info("provider.openai.unexpected_error", error=str(e))
        raise RuntimeError("Erro inesperado ao chamar a OpenAI.") from e

# ==============================
# app/openai_client.py
# Propósito:
# - Cliente OpenAI assíncrono para geração de respostas
# - Função 'judge' síncrona para atuar como juiz (JSON estrito)
# Notas:
# - Usa SDK openai>=1.x (classe OpenAI)
# - Model: definido por OPENAI_MODEL (default: gpt-4o-mini)
# ==============================
from __future__ import annotations

import os
import asyncio
from typing import Dict, Any, Optional

from openai import OpenAI


# --------------------------------------------
# Config
# --------------------------------------------
_OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") or None
_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client: Optional[OpenAI] = None
if _OPENAI_API_KEY:
    _client = OpenAI(api_key=_OPENAI_API_KEY)


def is_configured() -> bool:
    """Retorna True se a OPENAI_API_KEY estiver disponível."""
    return bool(_OPENAI_API_KEY and _client is not None)


# --------------------------------------------
# Helpers (sync) para chamar a API da OpenAI
# --------------------------------------------
def _oai_chat_sync(
    messages: list[dict[str, str]],
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    response_format: Optional[dict] = None,
) -> str:
    """
    Chamada síncrona ao endpoint de chat da OpenAI, retornando o texto.
    """
    if not is_configured():
        raise RuntimeError("OpenAI não configurado. Defina OPENAI_API_KEY.")

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        # Ex.: {'type': 'json_object'}
        kwargs["response_format"] = response_format

    resp = _client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
    content = resp.choices[0].message.content or ""
    return content.strip()


# --------------------------------------------
# API Async usada pelo app
# --------------------------------------------
async def ask_openai(
    prompt: str,
    *,
    model: str = _DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: float = 25.0,
) -> Dict[str, Any]:
    """
    Gera uma resposta com a OpenAI e retorna no formato:
      { "answer": "<texto>" }
    Lança exceção se não estiver configurado.
    """
    if not is_configured():
        raise RuntimeError("OpenAI não configurado. Defina OPENAI_API_KEY.")

    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(
                _oai_chat_sync,
                [{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
            ),
            timeout=timeout,
        )
        return {"answer": text}
    except asyncio.TimeoutError as te:
        raise RuntimeError("Timeout ao chamar OpenAI.") from te
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
    """
    Julga duas respostas conforme instruções do 'system' e 'user'.
    Retorna TEXTO cru (string). Quem extrai JSON é o judge.py.
    """
    if not is_configured():
        raise RuntimeError("OpenAI não configurado. Defina OPENAI_API_KEY.")

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response_format = {"type": "json_object"} if force_json else None
    # chamada síncrona
    return _oai_chat_sync(
        messages,
        model=_DEFAULT_MODEL,
        temperature=temperature,
        response_format=response_format,
    )

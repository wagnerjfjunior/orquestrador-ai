# ==============================
# app/gemini_client.py
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
    """
    Chamada síncrona ao Gemini. Usada dentro de asyncio.to_thread().
    Retorna apenas o texto.
    """
    mdl = genai.GenerativeModel(model)
    resp = mdl.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )
    # Alguns retornos podem vir em 'candidates' ou diretamente em 'text'
    text = getattr(resp, "text", None)
    if not text and hasattr(resp, "candidates") and resp.candidates:
        # fallback defensivo
        text = getattr(resp.candidates[0], "content", None)
        if hasattr(text, "parts") and text.parts:
            # junta partes de texto se necessário
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
    """
    Chamada síncrona ao Gemini com um 'system' informal (injetado no texto).
    Retorna texto cru (string).
    """
    # Como a SDK exposta aqui não usa o campo system_instruction,
    # injetamos o 'system' antes do 'user' para orientar o modelo.
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
    Gera uma resposta com o Gemini e retorna no formato:
      { "answer": "<texto>" }
    Lança exceção se não estiver configurado.
    """
    if not is_configured():
        raise RuntimeError("Gemini não configurado. Defina GEMINI_API_KEY.")

    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_gemini_generate_sync, prompt, model=model, temperature=temperature),
            timeout=timeout,
        )
        return {"answer": text}
    except asyncio.TimeoutError as te:
        raise RuntimeError("Timeout ao chamar Gemini.") from te
    except Exception as e:
        # Propaga a exceção para o handler do app
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

    Observação: aqui mantemos síncrono (como o judge.py espera).
    O timeout é melhor tratado pelo chamador (judge.py) se necessário.
    """
    if not is_configured():
        raise RuntimeError("Gemini não configurado. Defina GEMINI_API_KEY.")

    # O Gemini SDK é síncrono; esta função permanece síncrona.
    # Se precisar de timeout rígido síncrono, pode-se usar threads/sinais,
    # mas o orquestrador já trata exceções do juiz e aplica fallback.
    try:
        return _gemini_generate_with_system_sync(
            system=system,
            user=user,
            model=_DEFAULT_MODEL,
            temperature=temperature,
        )
    except Exception:
        # Deixa o judge.py lidar com parsing/fallback
        raise

    # --------------------------------------------------------------------
# Back-compat para a suíte de testes (append-only; não remove nada)
# --------------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional

@dataclass
class _SettingsShim:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or ""
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# objeto que os testes monkeypatcham: app.gemini_client.settings
settings = _SettingsShim()

def _shim_is_configured() -> bool:
    key = (
        (settings.GEMINI_API_KEY or "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )
    try:
        # se você mantiver uma global interna como _GEMINI_API_KEY
        key = key or ((_GEMINI_API_KEY or "").strip())  # noqa: F821
    except Exception:
        pass
    return bool(key)

# expõe o nome que os testes usam
is_configured = _shim_is_configured

async def ask(prompt: str, *, model: Optional[str] = None):
    """
    Alias compatível com a suíte de testes.
    Delegamos para sua função real (p.ex. ask_gemini) mantendo timeouts/parsing.
    """
    mdl = model or (settings.GEMINI_MODEL or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    # se sua função real for ask_gemini(...)
    return await ask_gemini(prompt, model=mdl)

# ------------------ TEST SHIM (append-only) ------------------
from dataclasses import dataclass
from typing import Optional

@dataclass
class _SettingsShim:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or ""
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# objeto que os testes monkeypatcham:
settings = _SettingsShim()

def is_configured() -> bool:
    key = (settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")).strip()
    return bool(key)

async def ask(prompt: str, *, model: Optional[str] = None):
    mdl = model or (settings.GEMINI_MODEL or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    # delega para SUA função real (que você já tem)
    return await ask_gemini(prompt, model=mdl)
# ------------------ /TEST SHIM ------------------


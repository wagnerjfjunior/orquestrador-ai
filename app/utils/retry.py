# app/utils/retry.py
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Type


class RetryExceededError(RuntimeError):
    """Lançado quando todas as tentativas de retry se esgotam."""
    pass


def retry(
    fn: Callable[[], Any],
    retries: int = 2,
    backoff_ms: int = 200,
    retry_on: Tuple[Type[BaseException], ...] = (TimeoutError, ConnectionError),
    sleep: Optional[Callable[[float], None]] = None,
) -> Any:
    """
    Executa `fn` com tentativas de retry em erros transitórios.

    - retries: novas tentativas após a primeira (total = 1 + retries)
    - backoff_ms: atraso (ms) entre tentativas; exponencial (x2)
    - retry_on: exceções que disparam retry
    - sleep: função que recebe segundos (permite no-op em testes)

    Retorna o valor de `fn` na primeira execução bem-sucedida
    ou lança o último erro após esgotar as tentativas.
    """
    attempts = 0
    delay_sec = max(backoff_ms, 0) / 1000.0

    while True:
        try:
            attempts += 1
            return fn()
        except retry_on as exc:
            if attempts > retries:
                raise RetryExceededError(f"Tentativas esgotadas ({attempts})") from exc
            if sleep:
                sleep(delay_sec)
            delay_sec = delay_sec * 2 if delay_sec > 0 else 0.0

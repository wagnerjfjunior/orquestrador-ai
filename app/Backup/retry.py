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

    - retries: número de novas tentativas após a primeira (total de chamadas = 1 + retries)
    - backoff_ms: atraso (milissegundos) entre tentativas, exponencial (x2) a cada falha
    - retry_on: tupla de exceções consideradas transitórias para retry
    - sleep: função de espera (recebe segundos). Se None, não dorme (útil para testes).

    Retorna o valor de `fn` na primeira execução bem-sucedida ou lança o último erro.
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
            # backoff exponencial simples
            delay_sec = delay_sec * 2 if delay_sec > 0 else 0.0

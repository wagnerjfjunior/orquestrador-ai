# Sprint 3 – Resiliência

import random
import time
from typing import Literal

CircuitState = Literal["closed", "open", "half_open"]

class CircuitBreaker:
    """
    CB por provider:
      - closed: tudo passa
      - open: bloqueia até passar reset_seconds
      - half_open: deixa 1 tentativa; sucesso -> closed, falha -> open
    """
    def __init__(self, fail_threshold: int = 5, reset_seconds: float = 30.0):
        self.fail_threshold = max(1, fail_threshold)
        self.reset_seconds = max(1.0, float(reset_seconds))

        self._state: CircuitState = "closed"
        self._fail_count = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == "open" and (time.time() - self._opened_at) >= self.reset_seconds:
            # janela de tentativa
            self._state = "half_open"
        return self._state

    def allow_request(self) -> bool:
        st = self.state
        if st == "open":
            return False
        # closed/half_open liberam; half_open será avaliado pelo resultado
        return True

    def record_success(self) -> None:
        # sucesso fecha e zera contador
        self._state = "closed"
        self._fail_count = 0
        self._opened_at = 0.0

    def record_failure(self) -> None:
        # falha: incrementa; em half_open volta a "open" e reinicia janela
        self._fail_count += 1
        if self._state == "half_open":
            self._open_circuit()
            return
        if self._fail_count >= self.fail_threshold:
            self._open_circuit()

    def _open_circuit(self) -> None:
        self._state = "open"
        self._opened_at = time.time()

def compute_backoff(attempt: int, base: float = 0.2, factor: float = 2.0, jitter: float = 0.1) -> float:
    """
    Exponential backoff com jitter proporcional:
      attempt=1 -> base
      attempt=2 -> base*factor
      ...
      jitter aplica +/- (jitter * valor_base)
    """
    attempt = max(1, int(attempt))
    delay = base * (factor ** (attempt - 1))
    if jitter > 0:
      delta = delay * jitter
      delay = random.uniform(max(0.0, delay - delta), delay + delta)
    return max(0.0, delay)

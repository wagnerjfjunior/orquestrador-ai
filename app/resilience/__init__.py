# Sprint 3 – Resiliência
# Placeholder: interface mínima para circuit breaker e backoff.
class CircuitBreaker:
    def __init__(self, fail_threshold: int = 5, reset_seconds: float = 30.0):
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_seconds
        # implementação virá em commits subsequentes

def compute_backoff(attempt: int, base: float = 0.2, factor: float = 2.0, jitter: float = 0.1) -> float:
    # implementação virá em commits subsequentes
    return max(base, base * (factor ** max(0, attempt - 1)))

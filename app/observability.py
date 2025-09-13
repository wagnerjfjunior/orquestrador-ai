# app/observability.py
import os
import sys
import time
import logging
import structlog
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Nível de log via variável de ambiente LOG_LEVEL (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _configure_logger():
    """Configura logger estruturado com structlog e stdlib logging."""
    logging.basicConfig(
        stream=sys.stdout,
        format="%(message)s",
        level=getattr(logging, LOG_LEVEL, logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("orquestrador-ai")


logger = _configure_logger()


class TraceMiddleware(BaseHTTPMiddleware):
    """
    Middleware de tracing/observabilidade:
    - Loga início e fim da request (estruturado)
    - Calcula tempo de resposta (ms)
    *Não* garante X-Request-ID (isso fica no RequestIDMiddleware).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response: Response | None = None
        try:
            logger.info("request.start", path=str(request.url), method=request.method)
            response = await call_next(request)
            return response
        finally:
            dur_ms = (time.perf_counter() - start) * 1000
            status = getattr(response, "status_code", None) if response is not None else None
            logger.info(
                "request.end",
                path=str(request.url),
                method=request.method,
                status=status,
                duration_ms=round(dur_ms, 2),
            )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Garante a presença e a propagação do X-Request-ID:
    - Lê do request (case-insensitive: aceita 'x-request-id' ou 'X-Request-ID')
    - Gera um ID se ausente
    - Escreve SEMPRE 'X-Request-ID' na resposta
    - Também expõe em request.state.request_id para outros componentes
    """

    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get(self.header_name) or request.headers.get(self.header_name.lower())
        request_id = incoming or os.urandom(6).hex()

        # Disponibiliza para quem precisar (ex.: logs, handlers)
        setattr(request.state, "request_id", request_id)

        response = await call_next(request)

        # GARANTE o header com a grafia exata que os testes exigem
        response.headers[self.header_name] = request_id
        return response


__all__ = ["logger", "TraceMiddleware", "RequestIDMiddleware"]

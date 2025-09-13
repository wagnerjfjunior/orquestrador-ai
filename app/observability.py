# app/observability.py
import os
import sys
import time
import logging
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Callable

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
    Middleware para traçar requisições:
    - gera request_id se não houver
    - loga início e fim da request
    - calcula tempo de resposta (ms)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id") or os.urandom(6).hex()

        logger.info("request.start", path=str(request.url), method=request.method, request_id=request_id)
        try:
            response = await call_next(request)
            return response
        finally:
            dur_ms = (time.perf_counter() - start) * 1000
            status = getattr(response, "status_code", None)
            logger.info(
                "request.end",
                path=str(request.url),
                method=request.method,
                status=status,
                duration_ms=round(dur_ms, 2),
                request_id=request_id,
            )

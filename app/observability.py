# app/observability.py
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from typing import Callable, Optional

import structlog
from structlog.contextvars import bind_contextvars, merge_contextvars, clear_contextvars
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# -------- Config de log / structlog --------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REQUEST_ID_HEADER = "X-Request-ID"

def _configure_logger():
    logging.basicConfig(
        stream=sys.stdout,
        format="%(message)s",
        level=getattr(logging, LOG_LEVEL, logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.stdlib.add_log_level,
            # injeta os contextvars (inclui request_id quando houver)
            merge_contextvars,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("orquestrador-ai")

logger = _configure_logger()

# -------- Middlewares --------
class TraceMiddleware(BaseHTTPMiddleware):
    """
    Observabilidade de requisições:
    - Loga início e fim
    - Calcula duração (ms)
    *Não* garante X-Request-ID (isso é do RequestIDMiddleware)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response: Optional[Response] = None
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
    Garante e propaga o X-Request-ID:
    - Lê do request (aceita 'X-Request-ID' ou 'x-request-id')
    - Gera UUID4 se ausente
    - Sempre escreve 'X-Request-ID' na resposta
    - Expõe em request.state.request_id
    - Faz bind no structlog contextvars pro ID aparecer nos logs
    """

    def __init__(self, app, header_name: str = REQUEST_ID_HEADER):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get(self.header_name) or request.headers.get(self.header_name.lower())
        request_id = incoming or str(uuid.uuid4())

        # Disponibiliza no state e no contexto do logger
        setattr(request.state, "request_id", request_id)
        bind_contextvars(request_id=request_id)

        response: Optional[Response] = None
        try:
            response = await call_next(request)
            return response
        finally:
            # Garante o header SEMPRE, mesmo em exceção
            if response is None:
                response = Response()
            response.headers[self.header_name] = request_id
            # Evita vazar contexto para a próxima request
            clear_contextvars()
            # Como estamos no finally, precisamos devolver a response
            # (o Starlette espera que retornemos a mesma instância criada aqui)
            # Portanto, só retornamos se não retornamos antes
            if not hasattr(response, "_already_returned"):  # flag defensiva
                response._already_returned = True  # type: ignore[attr-defined]
                return response

__all__ = ["logger", "TraceMiddleware", "RequestIDMiddleware"]

import os
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

# ---------- logger configurado em JSON ----------
def _configure_logger():
    # nível via env (default INFO)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            level=getattr(structlog, log_level, structlog.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()

logger = _configure_logger()

# ---------- middleware de trace ----------
class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()

        # bind do contexto
        log = logger.bind(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:
            status = 500
            log.error("unhandled_exception", error=str(exc))
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)

        # adiciona header para propagação
        response.headers["X-Trace-Id"] = trace_id

        # provider pode ser query param
        provider = request.query_params.get("provider")
        log = log.bind(
            status=status,
            latency_ms=elapsed_ms,
        )
        if provider:
            log = log.bind(provider=provider)

        # escreve log final de access
        log.info("http_request_done")

        return response

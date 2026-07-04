"""HTTP middleware for the web demo."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from restaurant_agent.logging_config import log_event

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to each request and response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()

        log_event(
            logger,
            logging.INFO,
            "request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            stage="request_start",
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            log_event(
                logger,
                logging.ERROR,
                "request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                stage="request_error",
                duration_ms=duration_ms,
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        log_event(
            logger,
            logging.INFO,
            "request completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            stage="request_complete",
            duration_ms=duration_ms,
        )
        return response

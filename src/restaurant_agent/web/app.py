"""FastAPI application factory for the web demo."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from restaurant_agent.logging_config import configure_web_logging, log_event
from restaurant_agent.web.middleware import RequestIdMiddleware
from restaurant_agent.web.routes import router
from restaurant_agent.web.security import generic_unexpected_error

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def create_app() -> FastAPI:
    configure_web_logging(os.getenv("LOG_LEVEL", "INFO"))

    app = FastAPI(
        title="Restaurant Agent Demo",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        detail = exc.detail
        if not isinstance(detail, str):
            detail = generic_unexpected_error()
        response = JSONResponse(
            status_code=exc.status_code, content={"message": detail}
        )
        request_id = _request_id(request)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        log_event(
            logger,
            logging.ERROR,
            "unhandled exception",
            request_id=_request_id(request),
            method=request.method,
            path=request.url.path,
            stage="unhandled_exception",
        )
        logger.exception(
            "Unhandled exception for %s %s", request.method, request.url.path
        )
        response = JSONResponse(
            status_code=500,
            content={"message": generic_unexpected_error()},
        )
        request_id = _request_id(request)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    return app


app = create_app()

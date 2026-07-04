"""HTTP routes for the web demo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from restaurant_agent.web.api_schemas import (
    MAX_CITY_LENGTH,
    MAX_NAME_LENGTH,
    GatherErrorResponse,
    GatherRequest,
)
from restaurant_agent.web.security import generic_unexpected_error, sanitize_user_message
from restaurant_agent.web.service import GatherRequestError, run_gather_for_web

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter()


def _parse_form_bool(value: str | None) -> bool:
    return value in {"true", "on", "1", "yes"}


def _validation_error_message(exc: ValidationError) -> str:
    for error in exc.errors():
        if error.get("type") == "missing":
            return "Restaurant name and city are required."
    return "Invalid request. Check name and city values."


def _empty_form_values() -> dict[str, Any]:
    return {
        "name": "",
        "city": "",
        "live_google": False,
    }


def _index_context(
    request: Request,
    *,
    result: Any = None,
    error: str | None = None,
    form: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "request": request,
        "result": result,
        "error": error,
        "form": form if form is not None else _empty_form_values(),
    }


@router.get("/", response_class=HTMLResponse)
async def landing_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        _index_context(request),
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/demo")
async def demo_redirect() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@router.post("/", response_class=HTMLResponse)
@router.post("/demo", response_class=HTMLResponse)
async def search_submit(
    request: Request,
    name: str = Form(..., max_length=MAX_NAME_LENGTH),
    city: str = Form(..., max_length=MAX_CITY_LENGTH),
    live_google: str | None = Form(None),
) -> HTMLResponse:
    form_values = {
        "name": name.strip(),
        "city": city.strip(),
        "live_google": _parse_form_bool(live_google),
    }

    try:
        gather_request = GatherRequest(**form_values)
    except ValidationError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            _index_context(request, error=_validation_error_message(exc), form=form_values),
            status_code=400,
        )

    try:
        result = run_gather_for_web(gather_request)
    except GatherRequestError as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            _index_context(request, error=exc.message, form=form_values),
            status_code=400,
        )
    except Exception:
        logger.exception("Unhandled error during demo gather")
        return templates.TemplateResponse(
            request,
            "index.html",
            _index_context(request, error=generic_unexpected_error(), form=form_values),
            status_code=500,
        )

    return templates.TemplateResponse(
        request,
        "index.html",
        _index_context(request, result=result, form=form_values),
    )


@router.post("/api/gather")
async def api_gather(request_body: GatherRequest) -> JSONResponse:
    try:
        result = run_gather_for_web(request_body)
        return JSONResponse(content=result.model_dump(mode="json"))
    except GatherRequestError as exc:
        return JSONResponse(
            status_code=400,
            content=GatherErrorResponse(message=exc.message).model_dump(),
        )
    except Exception:
        logger.exception("Unhandled error during API gather")
        return JSONResponse(
            status_code=500,
            content=GatherErrorResponse(message=generic_unexpected_error()).model_dump(),
        )

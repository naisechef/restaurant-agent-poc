"""Web service layer — runs one gather operation per request."""

from __future__ import annotations

import logging
import time

from restaurant_agent.claude_client import ClaudeClient
from restaurant_agent.config import (
    MissingAPIKeyError,
    MissingGooglePlacesAPIKeyError,
    Settings,
    load_settings,
)
from restaurant_agent.gather_graph import run_gather_graph
from restaurant_agent.logging_config import log_event
from restaurant_agent.schemas import GatherRunResult, RestaurantQuery
from restaurant_agent.sources.factory import build_adapters
from restaurant_agent.web.api_schemas import GatherRequest
from restaurant_agent.web.security import sanitize_gather_result, sanitize_user_message

logger = logging.getLogger(__name__)


class GatherRequestError(Exception):
    """Raised when a gather request cannot be fulfilled due to configuration."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def run_gather_for_web(
    request: GatherRequest,
    *,
    settings: Settings | None = None,
    request_id: str | None = None,
) -> GatherRunResult:
    """Execute exactly one gather operation and return a sanitized result."""
    settings = settings or load_settings()
    query = RestaurantQuery(name=request.name, city=request.city)
    started = time.perf_counter()

    log_event(
        logger,
        logging.INFO,
        "gather started",
        request_id=request_id,
        restaurant=request.name,
        city=request.city,
        stage="gather_start",
    )

    try:
        client = ClaudeClient(settings)
    except MissingAPIKeyError as exc:
        log_event(
            logger,
            logging.WARNING,
            "missing anthropic api key",
            request_id=request_id,
            restaurant=request.name,
            city=request.city,
            stage="config_error",
        )
        raise GatherRequestError(
            sanitize_user_message(str(exc))
            or "Live Claude API is not configured on the server."
        ) from exc

    live_source = "google" if request.live_google else None
    try:
        adapters = build_adapters(
            settings.fixtures_path,
            live_source=live_source,
            settings=settings,
        )
    except MissingGooglePlacesAPIKeyError as exc:
        log_event(
            logger,
            logging.WARNING,
            "missing google places api key",
            request_id=request_id,
            restaurant=request.name,
            city=request.city,
            stage="config_error",
        )
        raise GatherRequestError(
            sanitize_user_message(str(exc))
            or "Live Google Places is not configured on the server."
        ) from exc

    result = run_gather_graph(
        query,
        adapters,
        client,
        settings.confidence_threshold,
        dry_run=False,
        live_google=request.live_google,
    )

    result = result.model_copy(update={"dry_run": False, "live_google": request.live_google})
    sanitized = sanitize_gather_result(result)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    log_event(
        logger,
        logging.INFO,
        "gather completed",
        request_id=request_id,
        restaurant=request.name,
        city=request.city,
        stage="gather_complete",
        duration_ms=duration_ms,
    )
    return sanitized

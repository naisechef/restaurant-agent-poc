"""Local readiness checks for Cloud Run — no external network calls."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from restaurant_agent.config import load_settings

_LIVE_GOOGLE_REQUIRED_ENV = "LIVE_GOOGLE_REQUIRED"


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _check_present(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def evaluate_readiness() -> dict[str, Any]:
    """Return readiness status based on local configuration only."""
    settings = load_settings()
    live_google_required = _env_flag(_LIVE_GOOGLE_REQUIRED_ENV)

    fixtures_path = settings.fixtures_path
    fixtures_ok = fixtures_path.is_dir()

    anthropic_ok = _check_present(settings.anthropic_api_key)
    google_ok = _check_present(settings.google_places_api_key)

    checks: dict[str, Any] = {
        "anthropic_api_key": {"ok": anthropic_ok, "required": True},
        "google_places_api_key": {
            "ok": google_ok,
            "required": live_google_required,
        },
        "fixtures_path": {
            "ok": fixtures_ok,
            "required": True,
            "path": str(fixtures_path),
        },
    }

    ready = anthropic_ok and fixtures_ok
    if live_google_required:
        ready = ready and google_ok

    return {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }

"""Centralised logging configuration for the pipeline CLI and web demo."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per log line for Cloud Logging field indexing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "restaurant",
            "city",
            "stage",
            "duration_ms",
            "path",
            "method",
            "status_code",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a consistent timestamped format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_web_logging(level: str = "INFO") -> None:
    """Configure JSON logging on stdout for the web demo (Cloud Run friendly)."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)
    root.addHandler(handler)


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: Any,
) -> None:
    """Emit a structured log record with extra indexable fields."""
    logger.log(level, message, extra=fields)

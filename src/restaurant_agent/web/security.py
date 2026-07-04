"""Security helpers for the web demo — sanitize errors and redact sensitive data."""

from __future__ import annotations

import re
from typing import Any

from restaurant_agent.schemas import GatherRunResult, GraphNodeExecution

_PATH_LIKE = re.compile(
    r"(?:^|[\s\"'])"
    r"((?:/|\.\./|\./)[\w./-]+|"
    r"[A-Za-z]:\\[\w\\.-]+|"
    r"/(?:Users|home|var|tmp|app|data|outputs|docs)/[\w./-]+)"
)

_KNOWN_USER_MESSAGES = {
    "no evidence gathered from any source": "No evidence was found for this restaurant.",
}

_GENERIC_ERROR = "The gather operation could not be completed."
_GENERIC_UNEXPECTED = "An unexpected error occurred. Please try again."

_SENSITIVE_SUBSTRINGS = (
    "api_key",
    "api key",
    "anthropic",
    "secret",
    "token",
    "password",
    "credential",
    ".env",
)


def sanitize_user_message(message: str | None) -> str | None:
    """Return a user-safe error message with paths and secrets redacted."""
    if message is None:
        return None

    normalized = message.strip()
    if not normalized:
        return None

    lower = normalized.lower()
    for fragment in _SENSITIVE_SUBSTRINGS:
        if fragment in lower:
            return _GENERIC_ERROR

    for key, friendly in _KNOWN_USER_MESSAGES.items():
        if key in lower:
            return friendly

    redacted = _PATH_LIKE.sub(" [redacted] ", normalized)
    redacted = redacted.strip()
    if len(redacted) > 300:
        redacted = redacted[:297] + "..."
    return redacted or _GENERIC_ERROR


def _sanitize_graph_trace(
    trace: list[GraphNodeExecution] | None,
) -> list[GraphNodeExecution] | None:
    if trace is None:
        return None

    sanitized: list[GraphNodeExecution] = []
    for step in trace:
        update = _sanitize_trace_update(step.update)
        sanitized.append(
            GraphNodeExecution(
                node=step.node,
                update=update,
                summary=step.summary,
            )
        )
    return sanitized


def _sanitize_trace_update(update: dict[str, object]) -> dict[str, object]:
    clean: dict[str, object] = {}
    for key, value in update.items():
        if key in {"raw_text", "cleaned_text", "query"}:
            continue
        clean[key] = _sanitize_trace_value(value)
    return clean


def _sanitize_trace_value(value: object) -> object:
    if isinstance(value, str):
        return sanitize_user_message(value) or value
    if isinstance(value, list):
        return [_sanitize_trace_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(k): _sanitize_trace_value(v)
            for k, v in value.items()
            if str(k) not in {"raw_text", "cleaned_text"}
        }
    return value


def sanitize_gather_result(result: GatherRunResult) -> GatherRunResult:
    """Redact internal details before returning a result to clients."""
    return result.model_copy(
        update={
            "error": sanitize_user_message(result.error),
            "gather_errors": [
                msg
                for msg in (
                    sanitize_user_message(err) for err in result.gather_errors
                )
                if msg is not None
            ],
            "source_results": [
                sr.model_copy(update={"error": sanitize_user_message(sr.error)})
                for sr in result.source_results
            ],
            "graph_trace": _sanitize_graph_trace(result.graph_trace),
        }
    )


def generic_unexpected_error() -> str:
    return _GENERIC_UNEXPECTED

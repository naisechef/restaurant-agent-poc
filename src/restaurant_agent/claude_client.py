"""Thin Anthropic SDK wrapper — the only module that imports anthropic."""

from __future__ import annotations

import json
import re

import anthropic

from restaurant_agent.config import Settings, load_settings


class LLMRequestError(RuntimeError):
    """Raised when an Anthropic API request fails."""


def _model_supports_temperature(model: str) -> bool:
    """Return False for models that reject the temperature parameter."""
    return not model.startswith("claude-sonnet-5")


class ClaudeClient:
    """Send system+user prompts to Claude and return the raw text response."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        api_key = self._settings.require_api_key()
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=self._settings.request_timeout,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt pair and return the assistant's text content."""
        try:
            create_kwargs: dict[str, object] = {
                "model": self._settings.model,
                "max_tokens": self._settings.max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "timeout": self._settings.request_timeout,
            }
            if (
                self._settings.temperature is not None
                and _model_supports_temperature(self._settings.model)
            ):
                create_kwargs["temperature"] = self._settings.temperature

            message = self._client.messages.create(**create_kwargs)
        except anthropic.APIError as exc:
            raise LLMRequestError(f"Anthropic API request failed: {exc}") from exc

        if not message.content:
            raise LLMRequestError("Anthropic API returned an empty response.")

        first_block = message.content[0]
        text = getattr(first_block, "text", None)
        if not text:
            raise LLMRequestError("Anthropic API response did not contain text content.")

        return text


_EVIDENCE_MARKER = "Evidence:\n"
_OUTDOOR_HINTS = (
    "patio",
    "terrace",
    "outdoor",
    "deck",
    "rooftop",
    "al fresco",
    "garden terrace",
    "sidewalk",
)
_INDOOR_HINTS = (
    "indoor",
    "indoors",
    "inside",
    "basement",
    "no windows",
    "no exterior",
    "do not offer",
    "no outdoor",
    "strictly indoors",
)


def _extract_evidence_from_prompt(user_prompt: str) -> str:
    """Pull the evidence block from a built outdoor-seating user prompt."""
    start = user_prompt.find(_EVIDENCE_MARKER)
    if start == -1:
        return user_prompt.strip()

    text = user_prompt[start + len(_EVIDENCE_MARKER) :]
    for suffix in ("\n\nReturn JSON", "\nReturn JSON"):
        end = text.find(suffix)
        if end != -1:
            text = text[:end]
    return text.strip()


def _first_snippet(evidence: str) -> str:
    """Return a short verbatim snippet for dry-run JSON output."""
    if not evidence:
        return ""
    sentence_match = re.search(r"[^.!?]+[.!?]?", evidence)
    if sentence_match:
        return sentence_match.group(0).strip()
    return evidence[:120].strip()


class DryRunClaudeClient:
    """Deterministic local client for --dry-run; never calls the Anthropic API."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if "Your previous response could not be parsed" in user_prompt:
            return json.dumps(
                {
                    "label": "unknown",
                    "confidence": 0.5,
                    "evidence": [],
                    "reasoning": "Dry-run repair fallback returned unknown.",
                }
            )

        evidence = _extract_evidence_from_prompt(user_prompt)
        lower = evidence.lower()

        has_outdoor = any(hint in lower for hint in _OUTDOOR_HINTS)
        has_indoor = any(hint in lower for hint in _INDOOR_HINTS)
        explicit_no_outdoor = any(
            phrase in lower
            for phrase in (
                "do not offer",
                "do not have outdoor",
                "no patio",
                "no outdoor",
                "not offer patio",
            )
        )
        contradictory = (
            ("do not have outdoor" in lower or "no outdoor" in lower)
            and ("patio" in lower or "sidewalk" in lower)
            and "opens on" in lower
        )

        if contradictory:
            label, confidence = "unknown", 0.45
        elif explicit_no_outdoor or (has_indoor and not has_outdoor):
            label, confidence = "no", 0.91
        elif has_outdoor and not has_indoor:
            label, confidence = "yes", 0.92
        elif "weather permitting" in lower:
            label, confidence = "unknown", 0.55
        else:
            label, confidence = "unknown", 0.35

        snippet = _first_snippet(evidence)
        evidence_quotes = [snippet] if snippet and label != "unknown" else []

        return json.dumps(
            {
                "label": label,
                "confidence": confidence,
                "evidence": evidence_quotes,
                "reasoning": (
                    f"Dry-run heuristic classified the evidence as '{label}'."
                ),
            }
        )

"""Extraction agent — calls Claude and parses structured outdoor seating output."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from restaurant_agent.claude_client import LLMRequestError
from restaurant_agent.prompts.outdoor_seating import build_prompt
from restaurant_agent.schemas import ExtractionResult
from restaurant_agent.state import AgentState

if TYPE_CHECKING:
    from restaurant_agent.pipeline import LLMClient

logger = logging.getLogger(__name__)

_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    flags=re.DOTALL | re.IGNORECASE,
)

_REPAIR_PREFIX = """\
Your previous response could not be parsed as valid JSON.

Previous response:
"""

_REPAIR_EVIDENCE_PREFIX = """

Original evidence:
"""

_REPAIR_SUFFIX = """

Return ONLY a valid JSON object matching this schema — no markdown fences, no prose:
{
  "label": "yes" | "no" | "unknown",
  "confidence": <float between 0.0 and 1.0>,
  "evidence": [<list of exact verbatim snippets>],
  "reasoning": "<one or two sentences>"
}\
"""


def _build_repair_prompt(raw_response: str, evidence: str) -> str:
    """Build a repair user prompt without str.format on untrusted text."""
    return _REPAIR_PREFIX + raw_response + _REPAIR_EVIDENCE_PREFIX + evidence + _REPAIR_SUFFIX


def strip_markdown_fences(text: str) -> str:
    """Remove common markdown JSON code fences from a model response."""
    stripped = text.strip()
    match = _FENCE_PATTERN.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _parse_extraction_json(raw_text: str) -> ExtractionResult:
    """Parse and schema-validate a model response as ExtractionResult."""
    cleaned = strip_markdown_fences(raw_text)
    data = json.loads(cleaned)
    return ExtractionResult.model_validate(data)


def _apply_extraction(state: AgentState, result: ExtractionResult) -> AgentState:
    return state.model_copy(
        update={
            "prediction": result.label,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "reasoning": result.reasoning,
            "error": None,
        }
    )


def _apply_error(state: AgentState, message: str) -> AgentState:
    return state.model_copy(update={"error": message})


def run(state: AgentState, client: LLMClient) -> AgentState:
    """Build a prompt, call Claude, parse JSON, and populate extraction fields."""
    if state.cleaned_text is None:
        return _apply_error(state, "Missing cleaned_text: run preprocessing first.")

    system_prompt, user_prompt = build_prompt(state.cleaned_text)

    try:
        raw_response = client.complete(system_prompt, user_prompt)
    except LLMRequestError as exc:
        return _apply_error(state, str(exc))

    try:
        result = _parse_extraction_json(raw_response)
        return _apply_extraction(state, result)
    except (json.JSONDecodeError, ValidationError):
        logger.warning(
            "Malformed extraction JSON for %s; attempting one repair retry.",
            state.restaurant_id,
        )

    try:
        repair_response = client.complete(
            system_prompt,
            _build_repair_prompt(raw_response, state.cleaned_text),
        )
    except LLMRequestError as exc:
        return _apply_error(state, str(exc))

    try:
        result = _parse_extraction_json(repair_response)
        return _apply_extraction(state, result)
    except (json.JSONDecodeError, ValidationError) as repair_error:
        return _apply_error(
            state,
            f"Failed to parse extraction JSON after repair retry: {repair_error}",
        )

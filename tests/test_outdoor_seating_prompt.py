"""Unit tests for outdoor seating prompt construction."""

from __future__ import annotations

import json

from restaurant_agent.prompts.outdoor_seating import SYSTEM_PROMPT, build_prompt


def test_build_prompt_returns_system_and_user_prompts() -> None:
    evidence = "Patio seating with heaters available year-round."
    system_prompt, user_prompt = build_prompt(evidence)

    assert system_prompt == SYSTEM_PROMPT
    assert evidence in user_prompt
    assert "JSON only" in system_prompt
    assert '"yes"' in system_prompt
    assert '"no"' in system_prompt
    assert '"unknown"' in system_prompt


def test_build_prompt_requires_exact_evidence_snippets() -> None:
    _, user_prompt = build_prompt("Terrace open in summer.")

    assert "Terrace open in summer." in user_prompt
    assert "exact verbatim snippets" in SYSTEM_PROMPT.lower() or "exactly as they appear" in SYSTEM_PROMPT


def test_system_prompt_describes_json_schema_fields() -> None:
    for field in ("label", "confidence", "evidence", "reasoning"):
        assert field in SYSTEM_PROMPT

    # Prompt contract should align with ExtractionResult keys for downstream parsing.
    assert json.loads(
        '{"label": "unknown", "confidence": 0.4, "evidence": [], "reasoning": "No mention."}'
    )

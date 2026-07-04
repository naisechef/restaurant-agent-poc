"""Merge, deduplicate, and combine evidence from multiple sources."""

from __future__ import annotations

import re

from restaurant_agent.schemas import (
    Evidence,
    EvidenceSourceType,
    GatheredEvidence,
    SourceResult,
)

_RELIABILITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_SOURCE_TYPE_ORDER = {
    EvidenceSourceType.WEBSITE: 0,
    EvidenceSourceType.MAPS: 1,
    EvidenceSourceType.SEARCH: 2,
    EvidenceSourceType.REVIEW: 3,
}
_WHITESPACE_RE = re.compile(r"\s+")


def _normalise_snippet(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _evidence_sort_key(item: Evidence) -> tuple[int, int, str, str]:
    return (
        _RELIABILITY_ORDER[item.reliability],
        _SOURCE_TYPE_ORDER[item.source_type],
        item.source_name,
        _normalise_snippet(item.snippet),
    )


def merge_evidence(source_results: list[SourceResult]) -> GatheredEvidence:
    """Deduplicate snippets, aggregate errors, and order by reliability."""
    errors: list[str] = []
    seen: set[str] = set()
    items: list[Evidence] = []

    for result in source_results:
        if result.error:
            errors.append(f"{result.source_name}: {result.error}")

        for evidence in result.evidence:
            key = _normalise_snippet(evidence.snippet)
            if key in seen:
                continue
            seen.add(key)
            items.append(evidence)

    items.sort(key=_evidence_sort_key)
    return GatheredEvidence(items=items, errors=errors)


def combine_evidence_text(items: list[Evidence]) -> str:
    """Join evidence snippets into a single text block for preprocessing."""
    if not items:
        return ""

    sections: list[str] = []
    for item in items:
        header = f"[{item.source_type.value}]"
        if item.url:
            header = f"{header} {item.url}"
        sections.append(f"{header}\n{item.snippet}")

    return "\n\n".join(sections)

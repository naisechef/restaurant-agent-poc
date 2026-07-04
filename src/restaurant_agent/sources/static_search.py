"""Static fixture-based source adapter (no network)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from restaurant_agent.schemas import (
    Evidence,
    EvidenceReliability,
    EvidenceSourceType,
    RestaurantQuery,
)

logger = logging.getLogger(__name__)

_DEFAULT_RELIABILITY: dict[EvidenceSourceType, EvidenceReliability] = {
    EvidenceSourceType.WEBSITE: "high",
    EvidenceSourceType.SEARCH: "medium",
    EvidenceSourceType.MAPS: "medium",
    EvidenceSourceType.REVIEW: "low",
}


def _lookup_key(query: RestaurantQuery) -> str:
    return f"{query.name}|{query.city}".lower()


class StaticSearchAdapter:
    """Loads evidence snippets from a JSON fixture file keyed by name|city."""

    def __init__(
        self,
        name: str,
        fixtures_path: Path,
        source_type: EvidenceSourceType,
    ) -> None:
        self.name = name
        self._fixtures_path = fixtures_path
        self._source_type = source_type
        self._reliability = _DEFAULT_RELIABILITY[source_type]

    def gather(self, query: RestaurantQuery) -> list[Evidence]:
        if not self._fixtures_path.exists():
            logger.warning("Fixture file not found: %s", self._fixtures_path)
            return []

        raw = json.loads(self._fixtures_path.read_text(encoding="utf-8"))
        entries = raw.get(_lookup_key(query))
        if not entries:
            return []

        evidence: list[Evidence] = []
        for entry in entries:
            evidence.append(
                Evidence(
                    source_type=self._source_type,
                    source_name=self.name,
                    url=entry.get("url"),
                    snippet=entry["snippet"],
                    reliability=self._reliability,
                )
            )
        return evidence

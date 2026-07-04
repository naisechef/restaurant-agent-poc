"""Build source adapter instances from settings."""

from __future__ import annotations

from pathlib import Path

from restaurant_agent.schemas import EvidenceSourceType
from restaurant_agent.sources.base import SourceAdapter
from restaurant_agent.sources.fake import FakeSourceAdapter
from restaurant_agent.sources.static_search import StaticSearchAdapter


def build_adapters(
    fixtures_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, SourceAdapter]:
    """Return adapters keyed by role: website, search, reviews, maps.

    v1 uses static fixtures for search/reviews/maps and an empty website
    placeholder (no live scraping). The dry_run flag is accepted for API
    compatibility but does not change adapter selection.
    """
    _ = dry_run
    return {
        "website": FakeSourceAdapter("empty_website", evidence=[]),
        "search": StaticSearchAdapter(
            "static_search",
            fixtures_path / "search.json",
            EvidenceSourceType.SEARCH,
        ),
        "reviews": StaticSearchAdapter(
            "static_reviews",
            fixtures_path / "reviews.json",
            EvidenceSourceType.REVIEW,
        ),
        "maps": StaticSearchAdapter(
            "static_maps",
            fixtures_path / "maps.json",
            EvidenceSourceType.MAPS,
        ),
    }

"""Build source adapter instances from settings."""

from __future__ import annotations

from pathlib import Path

from restaurant_agent.config import Settings
from restaurant_agent.schemas import EvidenceSourceType
from restaurant_agent.sources.base import SourceAdapter
from restaurant_agent.sources.fake import FakeSourceAdapter
from restaurant_agent.sources.google_places import GooglePlacesAdapter
from restaurant_agent.sources.static_search import StaticSearchAdapter

_SUPPORTED_LIVE_SOURCES = frozenset({"google"})


def build_adapters(
    fixtures_path: Path,
    *,
    dry_run: bool = False,
    live_source: str | None = None,
    settings: Settings | None = None,
) -> dict[str, SourceAdapter]:
    """Return adapters keyed by role: website, search, reviews, maps.

    By default, search/reviews/maps use static fixtures and website is an
    empty placeholder (no live scraping). The dry_run flag is accepted for
    API compatibility but does not change adapter selection.

    When `live_source == "google"`, the "maps" role is replaced with a live
    `GooglePlacesAdapter`; `settings.require_google_places_api_key()` is
    called at this point, so a missing GOOGLE_PLACES_API_KEY only raises
    when Google is explicitly selected. All other roles are unaffected.
    """
    _ = dry_run

    adapters: dict[str, SourceAdapter] = {
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

    if live_source is None:
        return adapters

    if live_source not in _SUPPORTED_LIVE_SOURCES:
        raise ValueError(
            f"Unsupported live source: {live_source!r}. "
            f"Supported: {sorted(_SUPPORTED_LIVE_SOURCES)}"
        )

    if live_source == "google":
        if settings is None:
            raise ValueError("settings is required when live_source='google'")
        api_key = settings.require_google_places_api_key()
        adapters["maps"] = GooglePlacesAdapter(
            "google_places",
            api_key,
            timeout=settings.google_places_timeout,
            max_reviews=settings.google_places_max_reviews,
            max_review_chars=settings.google_places_review_snippet_chars,
        )

    return adapters

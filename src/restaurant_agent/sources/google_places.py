"""Live evidence adapter for the Google Places API (New).

Uses the official Text Search + Place Details endpoints only (no scraping of
Google Maps, Search, or Reviews pages). Exactly one Text Search request and
one Place Details request are issued per `gather()` call, with an explicit
FieldMask (never a wildcard) to keep cost and payload size bounded.

HTTP access is isolated behind the `PlacesTransport` protocol so tests never
touch the network: production code uses `HttpxPlacesTransport` (backed by a
shared `httpx.Client`), while tests inject a fake transport or an
`httpx.MockTransport`-backed client.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import httpx

from restaurant_agent.schemas import Evidence, EvidenceSourceType, RestaurantQuery

logger = logging.getLogger(__name__)

_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_PLACE_DETAILS_URL_TEMPLATE = "https://places.googleapis.com/v1/places/{place_id}"

# Explicit field lists only — never "*" — to bound response size and cost.
_SEARCH_FIELD_MASK = "places.id,places.displayName,places.formattedAddress"
_DETAILS_FIELD_MASK = (
    "id,displayName,formattedAddress,googleMapsUri,websiteUri,"
    "outdoorSeating,rating,userRatingCount,reviews,reviewSummary,editorialSummary"
)

_DEFAULT_TIMEOUT = 10.0
_DEFAULT_MAX_REVIEWS = 3
_DEFAULT_MAX_REVIEW_CHARS = 300


class GooglePlacesError(RuntimeError):
    """Raised when a Google Places API request fails.

    Caught by `run_adapter_safe()` in gather.py and converted into
    `SourceResult.error` — one failed live lookup never aborts the rest of
    the gather run.
    """


def _extract_localized_text(value: object) -> str | None:
    """Unwrap a Google Places 'LocalizedText'-shaped value into plain text.

    Candidate fields show up either as a bare string (`editorialSummary` in
    some API versions) or as `{"text": "...", "languageCode": "en"}`
    (`displayName`, `reviewSummary`, review `text`). This helper handles
    both shapes, including one level of nesting, without per-field logic.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        return _extract_localized_text(value.get("text"))
    return None


def _checkmark(flag: bool) -> str:
    return "\u2713" if flag else "\u2717"


class PlacesTransport(Protocol):
    """HTTP seam for the Google Places API — the only part tests mock."""

    def search_text(self, text_query: str, *, timeout: float) -> dict[str, object]: ...

    def get_place_details(
        self, place_id: str, *, timeout: float
    ) -> dict[str, object]: ...


class HttpxPlacesTransport:
    """Real Places API (New) transport backed by a shared `httpx.Client`."""

    def __init__(self, api_key: str, *, client: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client()

    def search_text(self, text_query: str, *, timeout: float) -> dict[str, object]:
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": _SEARCH_FIELD_MASK,
        }
        try:
            response = self._client.post(
                _TEXT_SEARCH_URL,
                headers=headers,
                json={"textQuery": text_query},
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GooglePlacesError(
                "Google Places Text Search failed with status "
                f"{exc.response.status_code}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise GooglePlacesError(
                f"Google Places Text Search request failed: {exc}"
            ) from exc
        return response.json()

    def get_place_details(self, place_id: str, *, timeout: float) -> dict[str, object]:
        headers = {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": _DETAILS_FIELD_MASK,
        }
        url = _PLACE_DETAILS_URL_TEMPLATE.format(place_id=place_id)
        try:
            response = self._client.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GooglePlacesError(
                "Google Places Place Details failed with status "
                f"{exc.response.status_code}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise GooglePlacesError(
                f"Google Places Place Details request failed: {exc}"
            ) from exc
        return response.json()


class GooglePlacesAdapter:
    """Evidence adapter backed by the official Google Places API (New).

    `gather()` is a thin orchestrator: one Text Search call, one Place
    Details call, then two focused builder methods turn the response into
    `Evidence`. Adding future candidate fields (opening hours, accessibility,
    payment methods, dog-friendly, parking, wheelchair access, ...) means
    extending `_build_place_evidence()` — or adding a new sibling builder —
    without growing a single monolithic parser.
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        *,
        transport: PlacesTransport | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_reviews: int = _DEFAULT_MAX_REVIEWS,
        max_review_chars: int = _DEFAULT_MAX_REVIEW_CHARS,
    ) -> None:
        self.name = name
        self._timeout = timeout
        self._max_reviews = max_reviews
        self._max_review_chars = max_review_chars
        self._transport = transport or HttpxPlacesTransport(api_key)

    def gather(self, query: RestaurantQuery) -> list[Evidence]:
        text_query = f"{query.name}, {query.city}"

        start = time.perf_counter()
        search_response = self._transport.search_text(text_query, timeout=self._timeout)
        search_latency_ms = (time.perf_counter() - start) * 1000

        candidates = search_response.get("places") or []
        logger.info(
            "Google Places text search for '%s' returned %d candidate(s) in %.0f ms",
            text_query,
            len(candidates),
            search_latency_ms,
        )
        if not candidates:
            logger.warning("Google Places found no candidates for '%s'", text_query)
            return []

        place_id = candidates[0].get("id")
        if not place_id:
            logger.warning(
                "Google Places top candidate for '%s' has no place id", text_query
            )
            return []

        details = self._transport.get_place_details(place_id, timeout=self._timeout)

        maps_url = details.get("googleMapsUri")
        website_url = details.get("websiteUri")

        place_evidence = self._build_place_evidence(details, maps_url, website_url)
        review_evidence = self._build_review_evidence(details, maps_url)

        self._log_summary(
            search_latency_ms=search_latency_ms,
            candidate_count=len(candidates),
            details=details,
            place_id=place_id,
            review_evidence=review_evidence,
        )

        return [*place_evidence, *review_evidence]

    def _build_place_evidence(
        self,
        details: dict[str, object],
        maps_url: str | None,
        website_url: str | None,
    ) -> list[Evidence]:
        """Convert place-level facts (outdoor seating, summaries) into MAPS evidence."""
        evidence: list[Evidence] = []

        outdoor_seating = details.get("outdoorSeating")
        if outdoor_seating is True:
            evidence.append(
                Evidence(
                    source_type=EvidenceSourceType.MAPS,
                    source_name=self.name,
                    url=maps_url,
                    snippet="Google Places lists outdoor seating as available.",
                    reliability="high",
                )
            )
        elif outdoor_seating is False:
            evidence.append(
                Evidence(
                    source_type=EvidenceSourceType.MAPS,
                    source_name=self.name,
                    url=maps_url,
                    snippet="Google Places does not list outdoor seating as available.",
                    reliability="high",
                )
            )
        # outdoorSeating absent from the response: never fabricate evidence.

        review_summary = _extract_localized_text(details.get("reviewSummary"))
        if review_summary:
            evidence.append(
                Evidence(
                    source_type=EvidenceSourceType.MAPS,
                    source_name=self.name,
                    url=maps_url,
                    snippet=review_summary,
                    reliability="medium",
                )
            )

        editorial_summary = _extract_localized_text(details.get("editorialSummary"))
        if editorial_summary:
            evidence.append(
                Evidence(
                    source_type=EvidenceSourceType.MAPS,
                    source_name=self.name,
                    # websiteUri is stored only as provenance here; it is never
                    # fetched or crawled (out of scope for this adapter).
                    url=website_url or maps_url,
                    snippet=editorial_summary,
                    reliability="medium",
                )
            )

        return evidence

    def _build_review_evidence(
        self,
        details: dict[str, object],
        maps_url: str | None,
    ) -> list[Evidence]:
        """Convert up to `max_reviews` review snippets into REVIEW evidence."""
        raw_reviews = details.get("reviews") or []
        evidence: list[Evidence] = []
        for raw_review in raw_reviews[: self._max_reviews]:
            text = _extract_localized_text(raw_review.get("text")) or _extract_localized_text(
                raw_review.get("originalText")
            )
            if not text:
                continue
            snippet = text[: self._max_review_chars].strip()
            evidence.append(
                Evidence(
                    source_type=EvidenceSourceType.REVIEW,
                    source_name=self.name,
                    url=maps_url,
                    snippet=snippet,
                    reliability="low",
                )
            )
        return evidence

    def _log_summary(
        self,
        *,
        search_latency_ms: float,
        candidate_count: int,
        details: dict[str, object],
        place_id: str,
        review_evidence: list[Evidence],
    ) -> None:
        """Emit an informational log block; never affects Evidence/CSV output."""
        display_name = _extract_localized_text(details.get("displayName")) or "unknown"
        rating = details.get("rating", "n/a")
        review_count = details.get("userRatingCount", "n/a")

        has_outdoor_seating = details.get("outdoorSeating") is not None
        has_editorial_summary = _extract_localized_text(details.get("editorialSummary")) is not None
        has_review_summary = _extract_localized_text(details.get("reviewSummary")) is not None

        log_lines = [
            "Google Places",
            "-------------",
            f"Search latency: {search_latency_ms:.0f} ms",
            "",
            f"Candidates returned: {candidate_count}",
            "",
            "Selected:",
            f"- Name: {display_name}",
            f"- Rating: {rating}",
            f"- Review count: {review_count}",
            f"- Place ID: {place_id}",
            "",
            "Evidence extracted:",
            f"{_checkmark(has_outdoor_seating)} outdoor seating",
            f"{_checkmark(has_editorial_summary)} editorial summary",
            f"{_checkmark(has_review_summary)} review summary",
            f"{_checkmark(bool(review_evidence))} {len(review_evidence)} review(s)",
        ]
        logger.info("\n".join(log_lines))

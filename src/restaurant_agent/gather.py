"""Parallel evidence gathering with per-adapter failure isolation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from restaurant_agent.schemas import RestaurantQuery, SourceResult
from restaurant_agent.sources.base import SourceAdapter

logger = logging.getLogger(__name__)


def run_adapter_safe(adapter: SourceAdapter, query: RestaurantQuery) -> SourceResult:
    """Run one adapter; capture failures as SourceResult.error instead of raising."""
    try:
        evidence = adapter.gather(query)
        place_location = getattr(adapter, "_place_location", None)
        return SourceResult(
            source_name=adapter.name,
            evidence=evidence,
            place_location=place_location,
        )
    except Exception as exc:
        logger.warning("Source adapter %s failed: %s", adapter.name, exc)
        return SourceResult(source_name=adapter.name, evidence=[], error=str(exc))


def gather_all(
    query: RestaurantQuery,
    adapters: list[SourceAdapter],
    *,
    max_workers: int = 4,
) -> list[SourceResult]:
    """Gather evidence from all adapters in parallel."""
    if not adapters:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_adapter_safe, adapter, query) for adapter in adapters]
        return [future.result() for future in futures]

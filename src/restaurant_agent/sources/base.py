"""Source adapter protocol for evidence gathering."""

from __future__ import annotations

from typing import Protocol

from restaurant_agent.schemas import Evidence, RestaurantQuery


class SourceAdapter(Protocol):
    """Common interface for evidence source adapters."""

    name: str

    def gather(self, query: RestaurantQuery) -> list[Evidence]: ...

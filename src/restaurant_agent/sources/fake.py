"""Fake source adapter for tests and dry-run."""

from __future__ import annotations

from restaurant_agent.schemas import Evidence, RestaurantQuery


class FakeSourceAdapter:
    """Returns canned evidence or raises a configured exception."""

    def __init__(
        self,
        name: str,
        evidence: list[Evidence] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        self.name = name
        self._evidence = list(evidence or [])
        self._raises = raises

    def gather(self, query: RestaurantQuery) -> list[Evidence]:
        if self._raises is not None:
            raise self._raises
        return list(self._evidence)

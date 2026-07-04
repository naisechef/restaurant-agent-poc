"""Evidence source adapters for parallel gathering."""

from restaurant_agent.sources.base import SourceAdapter
from restaurant_agent.sources.factory import build_adapters
from restaurant_agent.sources.fake import FakeSourceAdapter
from restaurant_agent.sources.google_places import GooglePlacesAdapter
from restaurant_agent.sources.static_search import StaticSearchAdapter

__all__ = [
    "FakeSourceAdapter",
    "GooglePlacesAdapter",
    "SourceAdapter",
    "StaticSearchAdapter",
    "build_adapters",
]

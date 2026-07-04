"""Tests for evidence merge and deduplication."""

from __future__ import annotations

from restaurant_agent.evidence_merge import combine_evidence_text, merge_evidence
from restaurant_agent.schemas import Evidence, EvidenceSourceType, SourceResult


def _evidence(
    snippet: str,
    *,
    source_type: EvidenceSourceType = EvidenceSourceType.SEARCH,
    source_name: str = "test",
    reliability: str = "medium",
    url: str | None = None,
) -> Evidence:
    return Evidence(
        source_type=source_type,
        source_name=source_name,
        snippet=snippet,
        reliability=reliability,  # type: ignore[arg-type]
        url=url,
    )


def test_merge_deduplicates_identical_snippets() -> None:
    results = [
        SourceResult(
            source_name="a",
            evidence=[_evidence("Outdoor patio seating.")],
        ),
        SourceResult(
            source_name="b",
            evidence=[_evidence("  outdoor   patio seating.  ")],
        ),
    ]
    merged = merge_evidence(results)
    assert len(merged.items) == 1


def test_merge_orders_by_reliability() -> None:
    results = [
        SourceResult(
            source_name="a",
            evidence=[
                _evidence("Low reliability.", reliability="low"),
                _evidence("High reliability.", reliability="high"),
                _evidence("Medium reliability.", reliability="medium"),
            ],
        ),
    ]
    merged = merge_evidence(results)
    assert [item.reliability for item in merged.items] == ["high", "medium", "low"]


def test_merge_orders_by_source_type_within_reliability() -> None:
    results = [
        SourceResult(
            source_name="a",
            evidence=[
                _evidence(
                    "Review snippet.",
                    source_type=EvidenceSourceType.REVIEW,
                    reliability="medium",
                ),
                _evidence(
                    "Search snippet.",
                    source_type=EvidenceSourceType.SEARCH,
                    reliability="medium",
                ),
                _evidence(
                    "Maps snippet.",
                    source_type=EvidenceSourceType.MAPS,
                    reliability="medium",
                ),
                _evidence(
                    "Website snippet.",
                    source_type=EvidenceSourceType.WEBSITE,
                    reliability="medium",
                ),
            ],
        ),
    ]
    merged = merge_evidence(results)
    assert [item.source_type for item in merged.items] == [
        EvidenceSourceType.WEBSITE,
        EvidenceSourceType.MAPS,
        EvidenceSourceType.SEARCH,
        EvidenceSourceType.REVIEW,
    ]


def test_merge_order_independent_of_input_order() -> None:
    maps = _evidence(
        "Outdoor seating: Yes.",
        source_type=EvidenceSourceType.MAPS,
        source_name="static_maps",
        reliability="medium",
    )
    search = _evidence(
        "Riverside terrace seating.",
        source_type=EvidenceSourceType.SEARCH,
        source_name="static_search",
        reliability="medium",
    )
    review = _evidence(
        "Loved the patio area.",
        source_type=EvidenceSourceType.REVIEW,
        source_name="static_reviews",
        reliability="low",
    )

    pipeline_order = [
        SourceResult(source_name="static_search", evidence=[search]),
        SourceResult(source_name="static_maps", evidence=[maps]),
        SourceResult(source_name="static_reviews", evidence=[review]),
    ]
    graph_order = [
        SourceResult(source_name="static_maps", evidence=[maps]),
        SourceResult(source_name="static_search", evidence=[search]),
        SourceResult(source_name="static_reviews", evidence=[review]),
    ]

    merged_pipeline = merge_evidence(pipeline_order)
    merged_graph = merge_evidence(graph_order)

    assert merged_pipeline.items == merged_graph.items
    assert [item.source_type for item in merged_pipeline.items] == [
        EvidenceSourceType.MAPS,
        EvidenceSourceType.SEARCH,
        EvidenceSourceType.REVIEW,
    ]


def test_merge_aggregates_errors() -> None:
    results = [
        SourceResult(source_name="broken", evidence=[], error="timeout"),
        SourceResult(source_name="also_broken", evidence=[], error="not found"),
    ]
    merged = merge_evidence(results)
    assert merged.errors == ["broken: timeout", "also_broken: not found"]
    assert merged.items == []


def test_combine_evidence_text_includes_source_headers() -> None:
    items = [
        _evidence(
            "Terrace seating.",
            source_type=EvidenceSourceType.WEBSITE,
            url="https://example.com",
        ),
        _evidence(
            "Nice patio.",
            source_type=EvidenceSourceType.REVIEW,
        ),
    ]
    combined = combine_evidence_text(items)
    assert "[website] https://example.com" in combined
    assert "Terrace seating." in combined
    assert "[review]" in combined


def test_combine_evidence_text_empty() -> None:
    assert combine_evidence_text([]) == ""

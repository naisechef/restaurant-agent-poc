"""Unit tests for preprocessing pure functions."""

from __future__ import annotations

from restaurant_agent.preprocessing import (
    clean_evidence,
    normalise_whitespace,
    strip_boilerplate,
)


def test_normalise_whitespace_collapses_spaces_and_blank_lines() -> None:
    raw = "  patio   seating  \r\n\r\n\n\n  with views  "
    assert normalise_whitespace(raw) == "patio seating\n\nwith views"


def test_normalise_whitespace_returns_empty_for_blank_input() -> None:
    assert normalise_whitespace("") == ""
    assert normalise_whitespace("   \n\t  ") == ""


def test_strip_boilerplate_removes_html_and_headers() -> None:
    raw = (
        "<p>Enjoy our <strong>patio</strong> seating.</p>\n"
        "MENU\n"
        "----------\n"
        "Outdoor tables available."
    )
    cleaned = strip_boilerplate(raw)
    assert "<p>" not in cleaned
    assert "MENU" not in cleaned
    assert "----------" not in cleaned
    assert "Enjoy our" in cleaned
    assert "patio" in cleaned
    assert "Outdoor tables available." in cleaned


def test_strip_boilerplate_decodes_html_entities() -> None:
    raw = "Fish &amp; chips on the&nbsp;terrace"
    assert strip_boilerplate(raw) == "Fish & chips on the terrace"


def test_clean_evidence_composes_strip_and_normalise() -> None:
    raw = "<div>  patio   seating  </div>\n\n\nMENU\n\n---\n\n  all year  "
    assert clean_evidence(raw) == "patio seating\n\nall year"


def test_clean_evidence_handles_empty_input() -> None:
    assert clean_evidence("") == ""

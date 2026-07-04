"""Pure text-cleaning functions for restaurant evidence preprocessing."""

from __future__ import annotations

import html
import re

_HTML_TAG_RE = re.compile(r"<[^>]+>", flags=re.IGNORECASE)
_MENU_HEADER_RE = re.compile(
    r"^\s*(?:MENU|OUR MENU|Navigation|Skip to content)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_SEPARATOR_LINE_RE = re.compile(r"^\s*[-=*_]{3,}\s*$", flags=re.MULTILINE)
_WHITESPACE_RUN_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace and normalise line endings."""
    if not text:
        return ""

    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE_RUN_RE.sub(" ", line).strip() for line in normalised.split("\n")]
    collapsed = "\n".join(lines)
    collapsed = _BLANK_LINES_RE.sub("\n\n", collapsed)
    return collapsed.strip()


def strip_boilerplate(text: str) -> str:
    """Remove HTML fragments and common repeated menu or navigation headers."""
    if not text:
        return ""

    without_tags = _HTML_TAG_RE.sub(" ", text)
    decoded = html.unescape(without_tags).replace("\u00a0", " ")
    without_headers = _MENU_HEADER_RE.sub("", decoded)
    without_separators = _SEPARATOR_LINE_RE.sub("", without_headers)
    return without_separators


def clean_evidence(text: str) -> str:
    """Apply boilerplate removal then whitespace normalisation."""
    return normalise_whitespace(strip_boilerplate(text))

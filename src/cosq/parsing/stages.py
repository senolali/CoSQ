"""Parsers for the CoSQ interrogation stages."""

from __future__ import annotations

import re

from cosq.types import Certainty

#: Leading list markers: "1.", "1)", "- ", "* ", "• ".
_BULLET = re.compile(r"^\s*(?:[-*•]|\(?\d{1,2}[.)])\s+")

#: Lines that are headers or refusals rather than list items.
_NOT_AN_ITEM = re.compile(
    r"^\s*(?:required information|information needed|sub-?facts?|needs?)\s*:?\s*$",
    re.IGNORECASE,
)

_MAX_NEEDS = 20

# Checked before the affirmative patterns, so "not certain" never matches "certain".
_UNCERTAIN = re.compile(
    r"\b(?:un(?:certain|sure)|not\s+(?:certain|sure)|don'?t\s+know|no)\b", re.IGNORECASE
)
_CERTAIN = re.compile(r"\b(?:certain|sure|confident|yes)\b", re.IGNORECASE)

#: First number in a reply: "85", "85%", "0.85", "Confidence: 90".
_NUMBER = re.compile(r"(\d+(?:\.\d+)?|\.\d+)\s*%?")


def parse_need_list(text: str, max_items: int = _MAX_NEEDS) -> list[str]:
    """Extract the sub-facts enumerated by CoSQ stage 1.

    Accepts numbered, dashed, or bulleted lists. Falls back to treating each
    non-empty line as an item when the model ignored the list format. Returns an
    empty list when nothing usable was produced — the caller decides what that
    means (see :class:`~cosq.decision.threshold.ThresholdRule`).

    ``max_items`` caps runaway generations; the cap is recorded rather than silently
    applied, because the sub-fact count feeds the abstention analysis [M7].
    """
    bulleted: list[str] = []
    plain: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _NOT_AN_ITEM.match(line):
            continue
        stripped = _BULLET.sub("", line).strip()
        if not stripped:
            continue
        if _BULLET.match(line):
            bulleted.append(stripped)
        else:
            plain.append(stripped)

    items = bulleted or plain
    # Deduplicate case-insensitively while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:max_items]


def parse_certainty(text: str) -> Certainty | None:
    """Map a stage-2 completion onto ``certain`` / ``uncertain``.

    Negations are tested first, so "I am not certain" resolves to ``uncertain``
    rather than matching the substring "certain". Returns ``None`` when the reply
    expresses neither, which the caller must handle explicitly rather than
    defaulting.
    """
    if not text.strip():
        return None
    if _UNCERTAIN.search(text):
        return "uncertain"
    if _CERTAIN.search(text):
        return "certain"
    return None


def parse_confidence(text: str) -> float | None:
    """Map a graded stage-2 reply onto a confidence in ``[0, 1]``.

    The prompt asks for an integer 0-100, but models drift, so several conventions are
    accepted. Disambiguation is explicit rather than guessed:

    * a value above 1 is a percentage — ``85`` and ``85%`` both give ``0.85``;
    * a value at or below 1 written with a decimal point is already a fraction —
      ``0.85`` gives ``0.85``;
    * a bare ``1`` is read as **1%**, not 100%, because the prompt asked for 0-100.
      Writing ``100`` is unambiguous; writing ``1`` for full confidence is not, and
      guessing in the model's favour would silently inflate coverage.

    When no number appears, falls back to :func:`parse_certainty` so a model that
    ignores the format and answers "Certain" still yields a usable value at the
    extremes. Returns ``None`` when neither is present — the caller must decide what
    that means rather than defaulting silently.
    """
    if not text.strip():
        return None

    match = _NUMBER.search(text)
    if match is not None:
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - the regex only matches numbers
            return None
        if value > 1.0:
            value /= 100.0
        elif "." not in raw:
            # A bare integer 0 or 1 on a 0-100 scale means 0% or 1%.
            value /= 100.0
        return min(max(value, 0.0), 1.0)

    certainty = parse_certainty(text)
    if certainty == "certain":
        return 1.0
    if certainty == "uncertain":
        return 0.0
    return None

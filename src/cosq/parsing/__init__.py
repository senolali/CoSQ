"""Parsers turn raw completions into structured values.

Parsers never call a model, and backends never parse. That separation is what makes
``records.jsonl`` re-scorable offline: fix a parser, re-score every condition, no GPU
and no re-querying.
"""

from cosq.parsing.answers import (
    ABSTENTION_MARKERS,
    detect_abstention,
    extract_answer_segment,
    match_open_answer,
    parse_mc_choice,
)
from cosq.parsing.stages import parse_certainty, parse_need_list

__all__ = [
    "ABSTENTION_MARKERS",
    "detect_abstention",
    "extract_answer_segment",
    "match_open_answer",
    "parse_certainty",
    "parse_mc_choice",
    "parse_need_list",
]

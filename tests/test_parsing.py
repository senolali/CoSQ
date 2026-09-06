import pytest

from cosq.parsing import (
    detect_abstention,
    extract_answer_segment,
    parse_certainty,
    parse_mc_choice,
    parse_need_list,
)

OPTIONS = ("four", "five", "more than five")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: B", 1),
        ("Answer: (C)", 2),
        ("Answer: five", 1),
        ("B) five", 1),
        ("Reasoning: hmm.\nAnswer: more than five", 2),
        ("The answer: five", 1),
        ("Answer: purple", None),
        ("", None),
    ],
)
def test_parse_mc_choice(text, expected):
    assert parse_mc_choice(text, OPTIONS) == expected


def test_reasoning_mentioning_an_option_does_not_override_the_stated_answer():
    # "four" appears in the reasoning; the stated answer is B.
    text = "Some sources say four, but that undercounts.\nAnswer: B"
    assert parse_mc_choice(text, OPTIONS) == 1


def test_ambiguous_mention_of_several_options_is_unparseable():
    assert parse_mc_choice("It could be four or five.", OPTIONS) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Certain", "certain"),
        ("Uncertain", "uncertain"),
        ("I am not certain.", "uncertain"),
        ("I'm sure about this", "certain"),
        ("Yes", "certain"),
        ("No", "uncertain"),
        ("banana", None),
        ("", None),
    ],
)
def test_parse_certainty(text, expected):
    assert parse_certainty(text) == expected


def test_negation_is_checked_before_the_affirmative():
    """ "not certain" must never match the substring "certain"."""
    assert parse_certainty("I am not certain") == "uncertain"
    assert parse_certainty("not sure at all") == "uncertain"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I don't know.", True),
        ("Answer: I do not know", True),
        ("Unknown", True),
        ("Answer: four", False),
        ("I don't know much about physics, but Answer: four", False),
        ("Reasoning: unknown things exist.\nAnswer: five", False),
        ("", False),
    ],
)
def test_detect_abstention(text, expected):
    assert detect_abstention(text) is expected


def test_hedging_is_not_abstention():
    """Counting hedges as abstention would inflate AR and flatter the method."""
    assert detect_abstention("I think it is probably five.") is False


def test_parse_need_list_formats():
    numbered = "Required information:\n1. the count\n2. the definition"
    dashed = "- the count\n- the definition"
    assert parse_need_list(numbered) == ["the count", "the definition"]
    assert parse_need_list(dashed) == ["the count", "the definition"]


def test_parse_need_list_deduplicates_and_caps():
    assert parse_need_list("1. same\n2. Same\n3. other") == ["same", "other"]
    many = "\n".join(f"{i}. item {i}" for i in range(1, 40))
    assert len(parse_need_list(many, max_items=5)) == 5


def test_parse_need_list_empty_when_unusable():
    assert parse_need_list("") == []
    assert parse_need_list("Required information:") == []


def test_extract_answer_segment_takes_the_last_cue():
    assert extract_answer_segment("Answer: one\nAnswer: two") == "two"

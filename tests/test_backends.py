import pytest

from cosq.backends.hf_local import HFLocalBackend
from cosq.backends.mock import MockBackend
from cosq.types import GenerationParams


def test_mock_is_deterministic(params):
    a, b = MockBackend(), MockBackend()
    prompt = "Question: x\n\nOptions:\nA) one\nB) two\n\nAnswer:"
    assert a.generate(prompt, params).text == b.generate(prompt, params).text


def test_mock_output_depends_on_the_prompt(params):
    backend = MockBackend()
    first = backend.generate("Options:\nA) one\nB) two\n\nAnswer:", params)
    second = backend.generate("Options:\nA) three\nB) four\n\nAnswer:", params)
    assert first.text != second.text or first.prompt != second.prompt


def test_mock_output_depends_on_generation_params():
    backend = MockBackend()
    prompt = "Options:\nA) one\nB) two\n\nAnswer:"
    a = backend.generate(prompt, GenerationParams(seed=1))
    b = backend.generate(prompt, GenerationParams(seed=2))
    assert (a.text, b.text) == (a.text, b.text)  # each is stable
    # Different seeds are allowed to coincide, but the key must differ.
    assert GenerationParams(seed=1).cache_key() != GenerationParams(seed=2).cache_key()


def test_canned_responses_win(params):
    backend = MockBackend(responses={"Answer:": "Answer: Z"})
    assert backend.generate("Answer:", params).text == "Answer: Z"


def test_completion_carries_provenance(params):
    completion = MockBackend(model_id="m", revision="r").generate("Answer:", params)
    assert (completion.model_id, completion.revision) == ("m", "r")
    assert completion.prompt == "Answer:"


def test_backend_returns_text_verbatim(params):
    backend = MockBackend(responses={"x": "  padded\n\n"})
    assert backend.generate("x", params).text == "  padded\n\n"


def test_hf_backend_refuses_a_floating_revision():
    """A moving revision silently changes what a run means."""
    for revision in ("main", "master", "HEAD", ""):
        with pytest.raises(ValueError, match="pinned commit SHA"):
            HFLocalBackend("meta-llama/Meta-Llama-3-8B-Instruct", revision)


def test_hf_backend_fingerprint_records_quantization():
    backend = HFLocalBackend("m", "abc123def")
    fingerprint = backend.fingerprint()
    assert fingerprint["revision"] == "abc123def"
    assert fingerprint["quantization"] == "nf4"  # mandatory, not a fallback [M1]


def test_a_wrapper_cannot_be_built_from_a_config():
    """CachingBackend decorates an already-built backend, so it is not selectable
    by name — and says so instead of failing obscurely."""
    from cosq.config import ModelConfig
    from cosq.runner.cache import CachingBackend

    config = ModelConfig(id="m", revision="r", backend="mock")
    with pytest.raises(NotImplementedError, match="cannot be built from a config"):
        CachingBackend.from_config(config)


def test_mock_answers_the_graded_stage_with_a_number(params):
    """Without this the graded pipeline abstains on everything under mock, and a smoke
    run would exercise none of it."""
    from cosq.parsing.stages import parse_confidence

    backend = MockBackend(confidence_range=(55, 98))
    prompt = "Consider this:\n\nalpha\n\nHow confident are you that this is correct?"
    value = parse_confidence(backend.generate(prompt, params).text)
    assert value is not None
    assert 0.55 <= value <= 0.98


def test_the_default_confidence_range_straddles_a_typical_threshold(params):
    """So a smoke run produces both answers and abstentions rather than one branch."""
    backend = MockBackend()
    values = [
        parse_conf
        for i in range(40)
        if (
            parse_conf := __import__(
                "cosq.parsing.stages", fromlist=["parse_confidence"]
            ).parse_confidence(
                backend.generate(f"q{i} How confident are you about this?", params).text
            )
        )
        is not None
    ]
    assert any(v < 0.8 for v in values) and any(v >= 0.8 for v in values)

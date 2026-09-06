import json

import pytest

from cosq.backends.mock import MockBackend
from cosq.config import ExperimentConfig
from cosq.data.base import sample_questions
from cosq.eval import compute_metrics, score_records
from cosq.report import analyze_run
from cosq.runner import run_experiment
from cosq.runner.cache import CompletionCache, cache_key
from cosq.runner.runner import load_records
from cosq.types import GenerationParams


def _config(**overrides):
    payload = {
        "name": "t",
        "model": {"id": "mock", "revision": "n/a", "backend": "mock", "quantization": None},
        "data": {"name": "jsonl", "path": "tests/fixtures/mini.jsonl", "n": 4, "seed": 1},
        "strategies": ["direct", "cot", "cot_abstain", "cosq"],
        "repeats": 2,
        "seed": 1002,
    }
    payload.update(overrides)
    return ExperimentConfig.from_mapping(payload)


def _run(tmp_path, **overrides):
    return run_experiment(
        _config(**overrides),
        results_root=tmp_path / "runs",
        cache_path=tmp_path / "cache.sqlite",
        backend=MockBackend(),
        allow_dirty=True,
    )


def test_run_writes_the_full_provenance_set(tmp_path):
    result = _run(tmp_path)
    files = {p.name for p in result.run_dir.iterdir()}
    assert {"config.resolved.yaml", "manifest.json", "records.jsonl"} <= files
    assert result.n_records == 4 * 4 * 2


def test_manifest_records_what_a_replication_needs(tmp_path):
    result = _run(tmp_path)
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["seed"] == 1002
    assert manifest["dataset"]["n"] == 4
    assert len(manifest["dataset"]["ids_sha256"]) == 64
    assert manifest["config_hash"]
    assert manifest["prompts_digest"]
    assert "python" in manifest["env"]
    assert "git" in manifest


def test_records_are_written_before_interpretation(tmp_path):
    """records.jsonl must carry the verbatim trace, so a run is re-scorable offline."""
    result = _run(tmp_path)
    rows = [
        json.loads(line) for line in (result.run_dir / "records.jsonl").read_text().splitlines()
    ]
    assert len(rows) == result.n_records
    cosq_rows = [row for row in rows if row["strategy"] == "cosq"]
    assert all(row["trace"] for row in cosq_rows)
    assert all("prompt" in turn and "completion" in turn for turn in cosq_rows[0]["trace"])


def test_records_round_trip(tmp_path):
    result = _run(tmp_path)
    reloaded = load_records(result.run_dir)
    assert len(reloaded) == result.n_records
    assert reloaded[0].question_id == result.records[0].question_id
    assert reloaded[0].trace == result.records[0].trace


def test_rerunning_is_served_entirely_from_cache(tmp_path):
    first = _run(tmp_path)
    assert first.cache_hits == 0
    second = _run(tmp_path)
    assert second.cache_misses == 0
    assert second.cache_hits > 0


def test_a_dirty_tree_is_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("cosq.runner.runner.git_state", lambda: {"commit": "x", "dirty": True})
    with pytest.raises(RuntimeError, match=r"uncommitted changes(.|\n)*--allow-dirty"):
        run_experiment(_config(), results_root=tmp_path, cache_path=None, backend=MockBackend())


def test_sampling_is_seeded_and_order_independent(questions):
    a, hash_a = sample_questions(questions, 3, seed=7)
    b, hash_b = sample_questions(list(reversed(questions)), 3, seed=7)
    assert [q.id for q in a] == [q.id for q in b]
    assert hash_a == hash_b


def test_sampling_refuses_to_over_draw(questions):
    with pytest.raises(ValueError, match="only"):
        sample_questions(questions, len(questions) + 1, seed=1)


def test_cache_key_separates_repeats():
    params = GenerationParams()
    assert cache_key("p", "m", "r", params, 0) != cache_key("p", "m", "r", params, 1)


def test_cache_marks_what_it_served(tmp_path):
    cache = CompletionCache(tmp_path / "c.sqlite")
    backend = MockBackend()
    completion = backend.generate("Answer:", GenerationParams())
    cache.put("k", completion)
    hit = cache.get("k")
    assert hit is not None and hit.cached is True and hit.text == completion.text
    cache.close()


def test_end_to_end_scoring_and_analysis(tmp_path):
    result = _run(tmp_path)
    questions = {q.id: q for q in result.questions}
    scored = score_records(result.records, questions)
    assert len(scored) == result.n_records

    metrics = compute_metrics([s for s in scored if s.strategy == "cosq"])
    assert 0.0 <= metrics.hallucination_rate <= 1.0
    assert metrics.counts.total == 4 * 2

    analysis = analyze_run(scored)
    assert set(analysis["strategies"]) == {"direct", "cot", "cot_abstain", "cosq"}
    assert "primary_h1" in analysis  # pre-registered contrast
    assert "secondary" in analysis  # CoSQ vs CoT+Abstain, Holm corrected
    assert "h2_coverage_matched" in analysis  # the test of the mechanism
    assert "power" in analysis["primary_h1"]


def test_scoring_rejects_an_unknown_question(tmp_path):
    result = _run(tmp_path)
    with pytest.raises(KeyError, match="no question"):
        score_records(result.records, {})


def test_h2_verdict_reports_uncertainty_not_a_bare_boolean(tmp_path):
    """A point comparison of two risks flips on noise at small answered counts, so the
    analysis must carry a paired test and a CI alongside it [M18]."""
    result = _run(tmp_path)
    questions = {q.id: q for q in result.questions}
    analysis = analyze_run(score_records(result.records, questions))
    h2 = analysis["h2_coverage_matched"]
    assert "verdict" in h2
    assert "matched_risk_difference" in h2
    assert "ci95" in h2["matched_risk_difference"]
    assert "matched_mcnemar" in h2
    assert isinstance(h2["advantage_is_significant"], bool)
    assert h2["n_questions_matched"] >= 0


def test_records_are_labelled_by_condition_not_class_name(tmp_path):
    """Two settings of one strategy must land in separate conditions."""
    config = _config(
        strategies=[
            "cot",
            {"name": "cosq_graded", "label": "graded_min", "aggregator": "minimum"},
            {"name": "cosq_graded", "label": "graded_mean", "aggregator": "mean"},
        ]
    )
    result = run_experiment(
        config,
        results_root=tmp_path / "runs",
        cache_path=None,
        backend=MockBackend(),
        allow_dirty=True,
    )
    labels = {r.strategy for r in result.records}
    assert labels == {"cot", "graded_min", "graded_mean"}

    scored = score_records(result.records, {q.id: q for q in result.questions})
    per_condition = {s.strategy for s in scored}
    assert per_condition == labels

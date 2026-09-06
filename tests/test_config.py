import pytest

from cosq.config import ExperimentConfig, ModelConfig, config_hash
from cosq.types import GenerationParams


def _config(**overrides):
    payload = {
        "name": "t",
        "model": {"id": "mock", "revision": "n/a", "backend": "mock", "quantization": None},
        "data": {"name": "jsonl", "path": "tests/fixtures/mini.jsonl", "n": 3, "seed": 1},
        "strategies": ["direct", "cosq"],
        "repeats": 2,
        "seed": 1002,
    }
    payload.update(overrides)
    return ExperimentConfig.from_mapping(payload)


def test_strategies_accept_names_or_mappings():
    config = _config(strategies=["direct", {"name": "cosq", "tau": 0.5}])
    assert [s.name for s in config.strategies] == ["direct", "cosq"]
    assert config.strategies[1].tau == 0.5


def test_unknown_keys_are_rejected_not_ignored():
    """A silently dropped key is a config that does not mean what it says."""
    with pytest.raises(ValueError, match="unknown keys"):
        _config(model={"id": "m", "revision": "r", "typo_field": 1})


def test_total_queries_is_workload_not_sample_size():
    config = _config()
    assert config.total_queries() == 3 * 2 * 2
    assert config.data.n == 3  # the statistical unit [M13]


def test_config_hash_is_stable():
    assert config_hash(_config()) == config_hash(_config())


def test_config_hash_changes_with_the_protocol():
    assert config_hash(_config()) != config_hash(_config(repeats=3))
    assert config_hash(_config()) != config_hash(
        _config(strategies=["direct", {"name": "cosq", "tau": 0.5}])
    )


def test_config_hash_covers_the_prompt_templates(monkeypatch):
    """Editing a prompt changes run identity, because a prompt is protocol [M2]."""
    from cosq import prompts

    before = config_hash(_config())
    monkeypatch.setattr(prompts, "prompt_digest", lambda lang="en": "different-digest")
    assert config_hash(_config()) != before


def test_generation_params_default_to_the_protocol():
    params = ModelConfig(id="m", revision="r").params()
    assert params == GenerationParams(temperature=0.0, top_p=0.95, max_new_tokens=256)


def test_pilot_config_matches_the_public_design():
    config = ExperimentConfig.load("configs/experiment/pilot_truthfulqa_open.yaml")
    assert [s.name for s in config.strategies] == [
        "direct",
        "cot",
        "cot_abstain",
        "cosq",
        "cosq_graded_gate",
    ]
    assert config.data.n == 100
    assert config.repeats == 1
    assert config.total_queries() == 500
    cosq = next(s for s in config.strategies if s.name == "cosq")
    assert cosq.tau == 0.0
    graded = next(s for s in config.strategies if s.name == "cosq_graded_gate")
    assert graded.threshold == 0.60
    assert graded.aggregator == "mean"


def test_duplicate_condition_names_are_rejected():
    """Two configurations of one strategy would be scored as a single condition, with
    no way to tell afterwards — so the config refuses to build."""
    with pytest.raises(ValueError, match="duplicate condition name"):
        _config(
            strategies=[
                {"name": "cosq_graded", "aggregator": "mean"},
                {"name": "cosq_graded", "aggregator": "minimum"},
            ]
        )


def test_a_label_distinguishes_two_settings_of_one_strategy():
    config = _config(
        strategies=[
            {"name": "cosq_graded", "label": "graded_mean", "aggregator": "mean"},
            {"name": "cosq_graded", "label": "graded_min", "aggregator": "minimum"},
        ]
    )
    assert [s.condition for s in config.strategies] == ["graded_mean", "graded_min"]


def test_condition_defaults_to_the_strategy_name():
    from cosq.config import StrategyConfig

    assert StrategyConfig(name="cosq").condition == "cosq"

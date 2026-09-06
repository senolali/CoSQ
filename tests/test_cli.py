import json

import pytest

from cosq.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "cosq" in capsys.readouterr().out


def test_list_shows_registered_plugins(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "cosq" in out and "mock" in out and "threshold" in out


def test_ask_with_the_mock_backend(capsys):
    assert (
        main(["ask", "How many senses?", "--model", "mock", "--option", "five", "--option", "six"])
        == 0
    )
    assert "Answer:" in capsys.readouterr().out


def test_ask_shows_the_interrogation_trace(capsys):
    main(["ask", "How many senses?", "--model", "mock", "--strategy", "cosq", "--show-trace"])
    out = capsys.readouterr().out
    assert "--- needs ---" in out
    assert "Decision:" in out


def test_run_evaluate_analyze_report(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("cosq.runner.runner.git_state", lambda: {"commit": "x", "dirty": False})
    config = "configs/experiment/smoke_mock.yaml"
    assert (
        main(
            [
                "run",
                "--config",
                config,
                "--backend",
                "mock",
                "--quiet",
                "--results-root",
                str(tmp_path / "runs"),
                "--cache",
                str(tmp_path / "c.sqlite"),
            ]
        )
        == 0
    )
    run_dir = capsys.readouterr().out.strip().splitlines()[-1]

    assert main(["evaluate", run_dir]) == 0
    assert "HR" in capsys.readouterr().out
    metrics = json.loads((tmp_path / "runs").glob("*/metrics.json").__next__().read_text())
    assert set(metrics) == {"direct", "cot", "cot_abstain", "cosq", "cosq_graded_gate_mean060"}

    assert main(["analyze", run_dir]) == 0
    capsys.readouterr()
    stats = json.loads((tmp_path / "runs").glob("*/stats.json").__next__().read_text())
    assert "h2_coverage_matched" in stats

    assert main(["report", run_dir, "--out", str(tmp_path / "out")]) == 0
    table = (tmp_path / "out" / "tables").glob("*.md").__next__().read_text()
    assert "Do not edit by hand" in table
    assert "| cosq |" in table

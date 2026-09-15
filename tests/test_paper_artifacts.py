from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "results"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_truthfulqa_public_panel_is_complete_and_internally_consistent():
    rows = _rows("truthfulqa_model_conditions.csv")
    models = {row["model_id"] for row in rows}
    conditions = {row["condition"] for row in rows}

    assert len(models) == 11
    assert len(conditions) == 17
    assert len(rows) == len(models) * len(conditions)
    assert "run_dir" not in rows[0]

    for row in rows:
        n = int(row["n"])
        correct = int(row["correct"])
        wrong = int(row["wrong"])
        abstained = int(row["idk"])
        unparseable = int(row["unparseable_count"])
        answered = correct + wrong

        assert n == 817
        assert correct + wrong + abstained + unparseable == n
        assert math.isclose(float(row["AA"]), correct / answered, abs_tol=1e-12)
        assert math.isclose(float(row["coverage"]), answered / n, abs_tol=1e-12)
        assert math.isclose(float(row["HR"]), wrong / n, abs_tol=1e-12)
        assert math.isclose(float(row["abstention_rate"]), abstained / n, abs_tol=1e-12)
        assert math.isclose(float(row["unparseable"]), unparseable / n, abs_tol=1e-12)


def test_truthfulqa_condition_summary_matches_the_public_model_panel():
    panel = _rows("truthfulqa_model_conditions.csv")
    summary = _rows("truthfulqa_condition_summary.csv")
    assert len(summary) == 17

    for aggregate in summary:
        rows = [row for row in panel if row["condition"] == aggregate["condition"]]
        assert len(rows) == 11
        assert int(aggregate["n_models"]) == 11
        for source_field, mean_field, sd_field in (
            ("AA", "AA_mean", "AA_sd"),
            ("coverage", "coverage_mean", "coverage_sd"),
            ("HR", "HR_mean", "HR_sd"),
            ("abstention_rate", "abstention_rate_mean", "abstention_rate_sd"),
            ("unparseable", "unparseable_mean", "unparseable_sd"),
        ):
            values = [float(row[source_field]) for row in rows]
            assert math.isclose(
                float(aggregate[mean_field]), statistics.fmean(values), abs_tol=1e-12
            )
            assert math.isclose(
                float(aggregate[sd_field]), statistics.stdev(values), abs_tol=1e-12
            )


def test_nq_public_panel_is_complete_and_metric_identity_holds():
    rows = _rows("nq_model_conditions.csv")
    assert len({row["model_id"] for row in rows}) == 5
    assert len({row["condition"] for row in rows}) == 5
    assert len(rows) == 25

    for row in rows:
        answered_accuracy = float(row["AA"])
        coverage = float(row["coverage"])
        risk = float(row["HR"])
        assert math.isclose(risk, coverage * (1.0 - answered_accuracy), abs_tol=1e-12)

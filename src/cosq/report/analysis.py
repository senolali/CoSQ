"""The pre-specified analysis, run over a completed experiment.

The hierarchy is fixed before data collection [M10]: one primary confirmatory test
(H1, uncorrected), a Holm-corrected secondary family, and everything else labelled
exploratory. This module encodes that hierarchy so it cannot drift at analysis time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from cosq.eval.metrics import compute_metrics
from cosq.eval.selective import coverage_matched_risk, random_abstention_risk, selective_point
from cosq.stats import (
    cliffs_delta,
    discordance_report,
    holm_bonferroni,
    mcnemar_exact,
    mcnemar_odds_ratio,
    risk_difference,
    wilcoxon_signed_rank,
)
from cosq.types import Question, ScoredRecord

PRIMARY_BASELINE = "cot"
PRIMARY_TREATMENT = "cosq"


def per_question_success(
    records: Sequence[ScoredRecord], strategy: str
) -> tuple[list[str], list[bool], list[float]]:
    """Collapse repeats to one value per question.

    Returns question ids, the majority-vote binary success used by McNemar, and the
    mean success rate across repeats used by Wilcoxon. Both are produced from the
    same records so the two tests are answering the same question [M4].
    """
    by_question: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        if record.strategy == strategy:
            by_question[record.question_id].append(record.outcome == "correct")
    ids = sorted(by_question)
    majority = [sum(by_question[qid]) * 2 > len(by_question[qid]) for qid in ids]
    means = [sum(by_question[qid]) / len(by_question[qid]) for qid in ids]
    return ids, majority, means


def _wrongness(records: Sequence[ScoredRecord], strategy: str) -> tuple[list[str], list[float]]:
    """Per-question hallucination indicator, averaged over repeats. HR's unit."""
    by_question: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.strategy == strategy:
            by_question[record.question_id].append(1.0 if record.outcome == "wrong" else 0.0)
    ids = sorted(by_question)
    return ids, [sum(by_question[qid]) / len(by_question[qid]) for qid in ids]


def _contrast(records: Sequence[ScoredRecord], treatment: str, baseline: str) -> dict[str, Any]:
    """One paired comparison, reported with every effect measure at once."""
    _, t_major, t_mean = per_question_success(records, treatment)
    _, b_major, b_mean = per_question_success(records, baseline)
    _, t_wrong = _wrongness(records, treatment)
    _, b_wrong = _wrongness(records, baseline)

    wrong_t = [value > 0.5 for value in t_wrong]
    wrong_b = [value > 0.5 for value in b_wrong]

    mcnemar = mcnemar_exact(wrong_t, wrong_b)
    wilcoxon = wilcoxon_signed_rank(t_wrong, b_wrong)
    point, ci = risk_difference(wrong_t, wrong_b)
    return {
        "treatment": treatment,
        "baseline": baseline,
        "mcnemar_exact": {"p": mcnemar.p_value, **mcnemar.detail},
        "wilcoxon": {"p": wilcoxon.p_value, **wilcoxon.detail},
        "cliffs_delta": cliffs_delta(t_wrong, b_wrong),
        "risk_difference": {"point": point, "ci95": list(ci)},
        "odds_ratio": mcnemar_odds_ratio(wrong_t, wrong_b),
        "power": asdict(discordance_report(wrong_t, wrong_b)),
        "accuracy_majority": {
            "treatment": sum(t_major) / len(t_major) if t_major else 0.0,
            "baseline": sum(b_major) / len(b_major) if b_major else 0.0,
        },
        "mean_success": {
            "treatment": sum(t_mean) / len(t_mean) if t_mean else 0.0,
            "baseline": sum(b_mean) / len(b_mean) if b_mean else 0.0,
        },
    }


def _matched_pair_test(
    treatment: Sequence[ScoredRecord], baseline: Sequence[ScoredRecord]
) -> dict[str, Any]:
    """Paired test of H2 on the questions the selective condition actually answered.

    Comparing two risk *point estimates* says nothing about whether the difference is
    real: at 12 answered questions the standard error on a risk near 0.17 is ~0.15, so
    a gap of 0.01 is indistinguishable from zero and the sign flips at random. This
    runs the proper paired test on the matched subset and returns a verdict that says
    so out loud [M18].
    """
    answered = {r.question_id for r in treatment if r.answered}
    if not answered:
        return {"verdict": "the selective condition answered nothing", "n_questions_matched": 0}

    def wrongness(records: Sequence[ScoredRecord]) -> dict[str, bool]:
        per_question: dict[str, list[bool]] = defaultdict(list)
        for record in records:
            if record.question_id in answered:
                per_question[record.question_id].append(record.outcome == "wrong")
        return {qid: sum(votes) * 2 > len(votes) for qid, votes in per_question.items() if votes}

    t_wrong, b_wrong = wrongness(treatment), wrongness(baseline)
    shared = sorted(set(t_wrong) & set(b_wrong))
    if not shared:
        return {"verdict": "no questions shared with the baseline", "n_questions_matched": 0}

    a = [t_wrong[q] for q in shared]
    b = [b_wrong[q] for q in shared]
    test = mcnemar_exact(a, b)
    point, ci = risk_difference(a, b)
    significant = test.p_value < 0.05

    if significant:
        direction = "lower" if point < 0 else "HIGHER"
        verdict = (
            f"at matched coverage the treatment's error rate is {direction} "
            f"(risk difference {point:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
            f"McNemar p={test.p_value:.4f}, n={len(shared)} questions)"
        )
    else:
        verdict = (
            f"INDISTINGUISHABLE at matched coverage: risk difference {point:+.3f}, "
            f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], McNemar p={test.p_value:.3f} over "
            f"{len(shared)} answered questions. The point comparison is not evidence "
            "either way."
        )
    return {
        "n_questions_matched": len(shared),
        "matched_risk_difference": {"point": point, "ci95": list(ci)},
        "matched_mcnemar": {"p": test.p_value, **test.detail},
        "advantage_is_significant": significant,
        "verdict": verdict,
    }


#: Conditions whose prompt asks for step-by-step reasoning. If their output is no
#: longer than the direct baseline's, the model ignored the instruction and the
#: manipulated variable was not actually manipulated [M21].
_REASONING_CONDITIONS = ("cot", "cot_abstain")
_REFERENCE_CONDITION = "direct"
#: Below this length ratio against the reference, a reasoning condition is flagged.
_COLLAPSE_RATIO = 1.2


def condition_diagnostics(scored: Sequence[ScoredRecord]) -> dict[str, Any]:
    """Check that the conditions actually differ in what the model produced.

    The independent variable is the questioning method, and the only evidence that it
    took effect is the generation itself. A model that ignores "think step by step"
    and emits a bare option turns `cot` into `direct` — the conditions collapse, the
    comparison compares nothing, and every downstream statistic still computes
    happily. This measures the final-stage output length per condition and says so
    when a reasoning condition is not visibly reasoning [M21].
    """
    lengths: dict[str, list[int]] = defaultdict(list)
    cued: dict[str, list[bool]] = defaultdict(list)
    for record in scored:
        answer_turns = [t for t in record.record.trace if t.stage == "answer"]
        if not answer_turns:
            continue
        completion = answer_turns[-1].completion
        lengths[record.strategy].append(len(completion))
        cued[record.strategy].append("answer:" in completion.casefold())

    per_condition = {
        name: {
            "mean_answer_chars": sum(values) / len(values),
            "answer_cue_rate": sum(cued[name]) / len(cued[name]),
            "n": len(values),
        }
        for name, values in sorted(lengths.items())
    }

    warnings: list[str] = []
    reference = per_condition.get(_REFERENCE_CONDITION)
    if reference and reference["mean_answer_chars"] > 0:
        for name in _REASONING_CONDITIONS:
            observed = per_condition.get(name)
            if observed is None:
                continue
            ratio = observed["mean_answer_chars"] / reference["mean_answer_chars"]
            observed["length_ratio_vs_direct"] = ratio
            if ratio < _COLLAPSE_RATIO:
                warnings.append(
                    f"{name!r} produced output only {ratio:.2f}x the length of "
                    f"{_REFERENCE_CONDITION!r} (mean {observed['mean_answer_chars']:.0f} vs "
                    f"{reference['mean_answer_chars']:.0f} chars). The model may have "
                    "ignored the step-by-step instruction, in which case this condition "
                    "is not distinct from the baseline and the contrast is empty [M21]."
                )
            if observed["answer_cue_rate"] < 0.5:
                warnings.append(
                    f"{name!r} emitted an explicit 'Answer:' line in only "
                    f"{observed['answer_cue_rate']:.0%} of completions, though its prompt "
                    "asks for one. Check the parser and the raw records before "
                    "interpreting this condition."
                )
    return {"per_condition": per_condition, "warnings": warnings}


def analyze_run(
    scored: Sequence[ScoredRecord], questions: Mapping[str, Question] | None = None
) -> dict[str, Any]:
    """Metrics per condition, the confirmatory hierarchy, and the H2 analysis."""
    by_strategy: dict[str, list[ScoredRecord]] = defaultdict(list)
    for record in scored:
        by_strategy[record.strategy].append(record)

    metrics = {
        name: compute_metrics(items).to_dict() for name, items in sorted(by_strategy.items())
    }
    selective = {
        name: asdict(selective_point(items, label=name))
        for name, items in sorted(by_strategy.items())
    }

    result: dict[str, Any] = {
        "n_questions": len({record.question_id for record in scored}),
        "strategies": sorted(by_strategy),
        "metrics": metrics,
        "selective": selective,
        # Ran first, because if the conditions collapsed nothing below it means
        # anything [M21].
        "condition_diagnostics": condition_diagnostics(scored),
    }

    # H1 -- the pre-registered primary test, reported uncorrected and alone.
    if PRIMARY_TREATMENT in by_strategy and PRIMARY_BASELINE in by_strategy:
        result["primary_h1"] = _contrast(scored, PRIMARY_TREATMENT, PRIMARY_BASELINE)

    # Secondary confirmatory family -- Holm corrected [M10].
    secondary: list[dict[str, Any]] = []
    if PRIMARY_TREATMENT in by_strategy and "cot_abstain" in by_strategy:
        secondary.append(_contrast(scored, PRIMARY_TREATMENT, "cot_abstain"))
    # The stage-3 ablation: cosq and cosq_gate gate on the identical signal and differ
    # only in how they generate once the rule says answer, so this contrast isolates
    # what stage 3 contributes [M19].
    if PRIMARY_TREATMENT in by_strategy and "cosq_gate" in by_strategy:
        ablation = _contrast(scored, PRIMARY_TREATMENT, "cosq_gate")
        ablation["measures"] = (
            "the cost or benefit of stage 3 alone: both conditions abstain on exactly "
            "the same questions, so any difference comes from constrained generation"
        )
        secondary.append(ablation)
    if secondary:
        adjusted = holm_bonferroni([item["mcnemar_exact"]["p"] for item in secondary])
        for item, (_, adj_p, reject) in zip(secondary, adjusted, strict=True):
            item["holm_adjusted_p"] = adj_p
            item["holm_reject"] = reject
        result["secondary"] = secondary

    # H2 -- the test of the mechanism: does the advantage survive coverage matching?
    if PRIMARY_TREATMENT in by_strategy and PRIMARY_BASELINE in by_strategy:
        treatment = by_strategy[PRIMARY_TREATMENT]
        baseline = by_strategy[PRIMARY_BASELINE]
        point = selective_point(treatment, label=PRIMARY_TREATMENT)
        try:
            matched = coverage_matched_risk(baseline, treatment)
            random_risk = random_abstention_risk(baseline, point.coverage, n_boot=200)
            paired = _matched_pair_test(treatment, baseline)
            result["h2_coverage_matched"] = {
                "cosq_coverage": point.coverage,
                "cosq_risk": point.risk,
                "cot_risk_on_same_questions": matched.risk,
                "cot_risk_random_abstention": random_risk,
                "risk_difference_vs_matched": point.risk - matched.risk,
                "advantage_survives_matching": point.risk < matched.risk,
                **paired,
                "note": (
                    "If the advantage does not survive coverage matching, the effect is "
                    "abstention volume rather than epistemic discrimination [M2]. "
                    "`advantage_survives_matching` is a bare point comparison and flips on "
                    "noise at small answered-question counts — read `verdict` instead [M18]."
                ),
            }
        except ValueError as exc:
            result["h2_coverage_matched"] = {"error": str(exc)}

    return result

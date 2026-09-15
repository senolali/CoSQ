# Paper artifacts

This directory contains the provider-neutral aggregate artifacts accompanying:

> Ali Şenol (2026), "When Should Large Language Models Abstain?
> Chain-of-Self-Questioning for Selective Risk Control."

The arXiv identifier is temporarily recorded as `xxxx.xxxx` and will be updated
after moderation.

## Protocol represented here

- TruthfulQA-MC1: 817 questions, 11 models, one repeat, balanced option
  positions with seed 1002.
- Conditions: forced-choice Direct and CoT plus Grounded, Critical, and
  Adaptive CoSQ at mean-confidence thresholds 0.50 through 0.90.
- NQ-Short: 300 questions and a five-model generalization panel at threshold
  0.90.
- `AA = correct / (correct + wrong)`.
- `coverage = (correct + wrong) / N`.
- `HR = wrong / N`, the unconditional wrong-commitment rate.

The CSV files are aggregate or condition-level outputs. They contain no API
credentials, private endpoints, provider configuration, response cache, or raw
private operational data.

## Files

- `results/truthfulqa_model_conditions.csv`: 11-model by 17-condition metrics.
- `results/truthfulqa_condition_summary.csv`: macro mean and standard deviation
  for each condition.
- `results/truthfulqa_paired_statistics.csv`: model-paired contrasts against CoT.
- `results/truthfulqa_abstention_quality.csv`: correct-abstention and lost-value
  summaries.
- `results/truthfulqa_baseline_compliance.csv`: strict one-label output audit for
  Direct and CoT.
- `results/nq_model_conditions.csv`: model-level NQ-Short generalization results.
- `figures/`: figures regenerated from these tables.

Regenerate every public figure from the repository root:

```bash
python -m pip install -e ".[paper]"
python scripts/plot_paper_results.py
```

SVG and PDF are produced by default. Use `--format svg` for SVG only.

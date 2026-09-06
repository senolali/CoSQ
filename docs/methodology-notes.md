# CoSQ Methodology Notes

CoSQ evaluates selective factual answering as a risk-coverage problem.

## Core protocol

1. A strategy receives the same benchmark question under the same model and decoding parameters.
2. The backend returns raw text only. It never parses, scores, or decides.
3. The runner writes every raw answer to `records.jsonl` before offline scoring.
4. Evaluation can be repeated from saved records without re-querying a model.

## Main metrics

- `answered_accuracy = correct / (correct + wrong)` for parseable committed answers.
- `coverage = (correct + wrong) / N`.
- `hallucination_rate = wrong / N`.
- `abstention_rate = idk / N`.
- `unparseable` is reported separately as measurement failure.

Answered accuracy must always be read together with coverage. A system can obtain high answered accuracy by answering only easy questions.

## CoSQ gates

Binary CoSQ uses a strict conjunctive gate: every required knowledge item must be judged `Certain`; otherwise the system abstains. Graded CoSQ replaces binary labels with 0-100 item-level confidence scores and compares an aggregate score, usually the mean, against a threshold.

## Reproducibility

- Prompt templates are versioned under `src/cosq/prompts/`.
- Config hashes include both experiment configuration and prompt templates.
- Model outputs are cached in SQLite when `--cache` is enabled.
- Hosted APIs are treated as unpinned model endpoints unless the provider exposes immutable revisions.

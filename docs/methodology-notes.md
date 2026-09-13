# CoSQ Methodology Notes

CoSQ treats factual answering as a selective-prediction problem. The framework
separates generation, parsing, scoring, and reporting so that a completed run
can be audited and rescored without contacting a model provider again.

## Protocol

1. Every condition receives the same question, model configuration, decoding
   parameters, and dataset sample.
2. A backend returns raw text only. It does not parse, score, or decide.
3. The runner stores raw completions and structured records in the run directory.
4. The evaluator maps committed answers to correct, wrong, abstained, or
   unparseable outcomes.
5. The analyzer produces paired summaries, risk-coverage quantities, and
   condition-level comparisons.

The question is the statistical unit. Repeated conditions are paired on question
ID, and the resolved configuration records the prompt digest, dataset settings,
model revision, and generation parameters.

## Three-stage CoSQ procedure

Stage 1 decomposes the question into the information items required for a
correct answer. The grounded and adaptive prompts may additionally label items
as CRITICAL or SUPPORTING.

Stage 2 asks for a bounded confidence score from 0 to 100 for each item. The
runner normalizes the score to [0, 1] and retains the original completion and
parsed values.

Stage 3 applies a threshold rule. Grounded-CoSQ uses accepted items and an
aggregate mean. Critical-CoSQ evaluates the critical subset. Adaptive-CoSQ
combines role-aware aggregate confidence with critical-item, minimum-confidence,
and consistency checks. A failed gate produces an abstention; a passed gate
produces a final answer from the accepted information.

## Metrics

For N questions, let C be correct committed answers, W wrong committed answers,
and A abstentions:

- AA = C / (C + W) is answered accuracy.
- Coverage = (C + W) / N.
- HR = W / N is the unconditional wrong-commitment rate.
- AR = A / N.
- R_answered = W / (C + W) = 1 - AA.

Unparseable output is reported separately as measurement failure. It is not
silently treated as a correct answer, wrong answer, or deliberate abstention.

Coverage is an operating characteristic, not an isolated objective. A
conservative threshold can intentionally route uncertain questions away from
answering. The relevant result is the joint HR-AA-coverage profile and its
behavior across thresholds.

## Reproducibility

- Prompt templates are versioned under src/cosq/prompts/.
- Configuration hashes include the resolved experiment and prompt digest.
- SQLite caching reuses identical completions and avoids duplicate calls.
- Run artifacts retain the resolved configuration, manifest, records, metrics,
  and analysis output.
- Hugging Face examples require immutable model commit revisions.
- Provider credentials belong in environment variables or an ignored .env
  file, never in YAML, source code, or committed run artifacts.

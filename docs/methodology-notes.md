# CoSQ Methodology Notes

CoSQ treats factual answering as a selective-prediction problem. The framework
separates generation, parsing, scoring, and reporting so that a completed run
can be audited and rescored without contacting a model provider again.

## Protocol

1. Every condition receives the same question, model configuration, decoding
   parameters, and dataset sample.
2. Multiple-choice experiments can deterministically rebalance correct option
   positions. The option seed and hash of the exact presentation are stored in
   the run manifest.
3. A backend returns raw text only. It does not parse, score, or decide.
4. The runner stores raw completions and structured records in the run directory.
5. The evaluator maps committed answers to correct, wrong, abstained, or
   unparseable outcomes.
6. The analyzer produces paired summaries, risk-coverage quantities, and
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
parsed values. Structured Grounded-CoSQ output is parsed only from its labelled
`CONFIDENCE:` field. A missing or malformed score follows the configured
conservative fallback and is counted for audit.

Stage 3 applies a threshold rule. Grounded-CoSQ aggregates confidence across
the required information items and uses threshold-accepted facts to construct
the final answer. Critical-CoSQ evaluates the critical subset. Adaptive-CoSQ
combines role-aware aggregate confidence with critical-item, minimum-confidence,
and consistency checks. A failed gate produces an abstention; a passed gate
produces a final answer from the accepted information.

The paper protocol uses two forced-choice controls and three CoSQ variants at
mean-confidence thresholds 0.50, 0.60, 0.70, 0.80, and 0.90, for 17 conditions
in total. Direct and CoT must return exactly one option label and are not offered
abstention. A malformed forced-choice response is scored as wrong.

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
- Manifests retain both the sampled-question hash and exact option-presentation
  hash, allowing offline evaluation to reconstruct the same answer key.
- Hugging Face examples require immutable model commit revisions.
- Provider credentials belong in environment variables or an ignored .env
  file, never in YAML, source code, or committed run artifacts.

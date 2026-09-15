# Changelog

All notable changes to CoSQ are documented here.

## 0.2.1 - 2026-09-15

- Add deterministic, balanced multiple-choice option placement and record the
  exact presentation hash in every run manifest.
- Enforce true forced-choice Direct and CoT controls; malformed baseline output
  is scored as wrong rather than converted into an implicit abstention.
- Parse Grounded-CoSQ confidence only from the labelled `CONFIDENCE:` field so
  numbers appearing inside generated facts cannot be mistaken for scores.
- Add the 30-question pilot and complete 817-question, 17-condition
  TruthfulQA-MC1 paper configurations.
- Add aggregate paper results, reproducible figure generation, citation
  metadata, and a tested PyPI Trusted Publishing workflow.

## 0.2.0 - 2026-09-12

- Add Grounded-CoSQ, Critical-CoSQ, and Adaptive-CoSQ strategies.
- Add local Hugging Face and OpenAI-compatible execution examples.
- Add PyPI packaging and GitHub Actions publishing.

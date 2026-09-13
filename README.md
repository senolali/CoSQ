# CoSQ: Chain-of-Self-Questioning

[![PyPI](https://img.shields.io/pypi/v/cosq.svg)](https://pypi.org/project/cosq/)
[![Python](https://img.shields.io/pypi/pyversions/cosq.svg)](https://pypi.org/project/cosq/)
[![Tests](https://github.com/senolali/cosq/actions/workflows/ci.yml/badge.svg)](https://github.com/senolali/cosq/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CoSQ is a reproducible framework for **selective factual answering**. Before an
LLM commits to an answer, it decomposes the question into required information,
estimates confidence for each item, and either produces an answer or abstains.
The goal is not to answer every question. It is to make answer commitment an
explicit, measurable risk-control decision.

This repository is the public research release accompanying:

> **When Should LLMs Abstain? Chain-of-Self-Questioning for Selective Risk Control**

## Why CoSQ?

In high-consequence settings, an incorrect answer can be more harmful than a
referral. A clinical decision-support system, for example, should sometimes
direct a case to a specialist rather than present an unsupported answer with
unwarranted confidence. CoSQ makes that behavior inspectable and tunable through
the standard risk-coverage perspective used in selective prediction.

## Method

CoSQ has three stages:

1. **Information decomposition.** The model lists the facts needed to answer the
   question correctly. Grounded and adaptive variants also assign a role to each
   item, such as `CRITICAL` or `SUPPORTING`.
2. **Item-level confidence.** The model gives a confidence score from 0 to 100
   for each item. Scores are normalized to the interval `[0, 1]` by the runner.
3. **Selective commitment.** A decision rule aggregates the item scores. If the
   gate passes, the model answers using the accepted information; otherwise the
   system returns an explicit abstention.

The public implementation keeps raw completions and parsed records separate so
that scoring and analysis can be repeated without making another model call.

### CoSQ variants

- **Grounded-CoSQ** answers from information items whose confidence passes the
  threshold. The gate uses the mean confidence of the accepted items.
- **Critical-CoSQ** asks the model to distinguish critical from supporting
  information and applies the gate to the critical items.
- **Adaptive-CoSQ** combines role-aware aggregate confidence with critical-item,
  minimum-confidence, and consistency checks. It is intended as a flexible
  operating policy when different questions require different amounts of
  information.

The baselines included in the package are direct answering, chain-of-thought
answering, and abstention-aware chain-of-thought answering. Prompts are versioned
text resources shipped with the package.

## Metrics

Let `N` be the number of questions, `C` correct committed answers, `W` wrong
committed answers, and `A` abstentions.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Answered accuracy (AA) | `C / (C + W)` | Accuracy among committed answers |
| Coverage | `(C + W) / N` | Fraction of questions answered |
| Hallucination rate (HR) | `W / N` | Unconditional wrong-commitment rate |
| Abstention rate (AR) | `A / N` | Fraction routed away from answering |
| Answered risk | `W / (C + W) = 1 - AA` | Error rate conditional on commitment |

Coverage is reported as an operating characteristic, not as a standalone
objective. A conservative system can deliberately answer fewer questions when
wrong answers carry a high downstream cost. The useful comparison is the joint
behavior of HR, AA, and coverage across thresholds.

Unparseable outputs are tracked separately as a measurement-quality indicator.
They are not silently counted as correct or wrong answers.

## Installation

From PyPI:

```bash
python -m pip install cosq
```

From source, including development tools:

```bash
git clone https://github.com/senolali/cosq.git
cd cosq
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS/Linux
# source .venv/bin/activate

python -m pip install -e ".[dev]"
```

Optional model integrations:

```bash
python -m pip install -e ".[hf]"      # local Hugging Face models
python -m pip install -e ".[openai]"  # OpenAI API models
```

## Quick start without an API

The mock backend exercises the complete runner, cache, evaluator, analyzer, and
reporting path without a network request or model credential:

```bash
cosq run --config configs/experiment/smoke_mock.yaml --backend mock --allow-dirty
```

The command prints the generated run directory. Use that directory for offline
evaluation and reporting:

```bash
cosq evaluate results/runs/<RUN_DIR>
cosq analyze results/runs/<RUN_DIR>
cosq report results/runs/<RUN_DIR>
```

Run `cosq --help` and `cosq <command> --help` for the complete CLI reference.

## Running a local Hugging Face model

Edit a model file under `configs/model/` and replace the placeholder revision
with an immutable Hugging Face commit SHA. This makes the experiment traceable
and avoids silently changing model weights. For example:

```yaml
id: meta-llama/Meta-Llama-3-8B-Instruct
revision: "REPLACE-WITH-HUGGINGFACE-COMMIT-SHA"
backend: hf_local
quantization: nf4
compute_dtype: bfloat16
generation:
  temperature: 0.0
  top_p: 0.95
  max_new_tokens: 256
  seed: 1002
backend_params:
  device_map: auto
  trust_remote_code: false
```

Set `HF_TOKEN` only when the selected model is gated. Then run the public
experiment template:

```bash
cosq run --config configs/experiment/pilot_truthfulqa_open.yaml --allow-dirty
```

The example uses TruthfulQA MC1 and includes direct, CoT, Grounded-CoSQ,
Critical-CoSQ, and Adaptive-CoSQ conditions at the illustrative `0.90`
threshold. For a publication-scale study, define the full threshold sweep in a
separate configuration and record the resolved YAML with the run.

## Running an OpenAI API model

Set the provider credential in the process environment or in a local `.env`
file, which is ignored by Git:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-key"

# macOS/Linux
# export OPENAI_API_KEY=your-key
```

Use `configs/model/openai_gpt4o_mini.yaml` as a starting point and review its
model identifier, revision, and generation settings before running:

```bash
cosq run --config configs/experiment/pilot_truthfulqa_open_openai.yaml --allow-dirty
```

Credentials are read at runtime and are never part of a resolved experiment
configuration or a committed result.

## Caching and offline analysis

Pass a SQLite cache to reuse identical completions:

```bash
cosq run --config configs/experiment/pilot_truthfulqa_open.yaml \
  --cache results/cache.sqlite --allow-dirty
```

The cache is keyed by the model configuration, prompt version, generation
parameters, and prompt text. Re-running an unchanged condition reuses its raw
completion rather than calling the provider again. Run artifacts contain the
resolved configuration, manifest, records, and derived metrics; generated
results and SQLite databases are intentionally ignored by Git.

## Datasets

The package provides adapters for:

- `jsonl`: local JSONL datasets for custom experiments;
- `truthfulqa_mc1`: TruthfulQA multiple-choice evaluation;
- `nq_short`: short-answer Natural Questions evaluation;
- `mmlu`: MMLU subject or aggregate evaluation when the optional dataset
  dependencies are installed.

For custom data, inspect `src/cosq/data/jsonl.py` and the fixture under
`tests/fixtures/`. Keep dataset sampling seeds and the sampled question IDs in
the run manifest so that the statistical unit remains the question.

## Development and verification

```bash
python -m pytest
python -m ruff check .
python -m build
python -m twine check dist/*
```

The repository intentionally contains no provider credentials, private
endpoints, operational secrets, or generated experimental results. Copy
`.env.example` to `.env` for local credentials and never commit the resulting
file.

## Maintainer release workflow

PyPI releases are published through GitHub Actions using PyPI Trusted
Publishing. No long-lived PyPI token is stored in the repository or in GitHub
secrets. To publish a release, configure the repository as a trusted publisher
for the `cosq` project on PyPI, using owner `senolali`, repository `CoSQ`, and
workflow file `.github/workflows/publish.yml`. Then either publish a GitHub
Release or start the `Publish package to PyPI` workflow manually. The package
version is read from `pyproject.toml`; update it before publishing a new
release.

## Repository layout

```text
configs/                 Public experiment and model examples
docs/                    Method and reproducibility notes
src/cosq/backends/       Mock, local Hugging Face, and OpenAI backends
src/cosq/data/           Dataset adapters
src/cosq/strategies/     Baselines and CoSQ variants
src/cosq/runner/         Execution, caching, and manifests
src/cosq/eval/           Metrics and scoring
src/cosq/report/         Tables and analysis reports
tests/                   Unit tests and a network-free fixture
```

## Citation and links

- GitHub: <https://github.com/senolali/cosq>
- PyPI: <https://pypi.org/project/cosq/>

If you use the framework, please cite the accompanying paper and report the
package version, model revision, dataset version, threshold, and cache policy.

## License

CoSQ is released under the MIT License. See [LICENSE](LICENSE).

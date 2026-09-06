# CoSQ: Chain-of-Self-Questioning

CoSQ is a Python framework for **selective factual answering** with large language models. Instead of forcing a model to answer every question, CoSQ asks the model to decompose the question into required knowledge items, assess whether those items are supported, and then either answer or abstain with `I don't know`.

The framework is designed for reproducible experiments on hallucination, abstention, answered accuracy, and risk-coverage trade-offs.

## Why CoSQ?

Standard accuracy rewards guessing: an abstention is usually scored the same as a wrong answer. That is a poor fit for settings where a wrong answer is more costly than saying "I don't know". CoSQ treats factual answering as a selective prediction problem and reports:

- **Answered accuracy**: accuracy among parseable committed answers.
- **Coverage**: the fraction of questions the model answers.
- **Hallucination rate**: wrong committed answers divided by all questions.
- **Abstention rate**: explicit `I don't know` responses.
- **Unparseable rate**: outputs that cannot be mapped to the benchmark answer space.

## Installation

From PyPI, after release:

```bash
pip install cosq
```

For local development:

```bash
git clone https://github.com/senolali/cosq.git
cd cosq
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Optional backends:

```bash
pip install -e ".[hf]"      # local Hugging Face / transformers models
pip install -e ".[openai]"  # OpenAI API backend
```

## Quick Start

Run a no-network smoke test with the deterministic mock backend:

```bash
cosq run --config configs/experiment/smoke_mock.yaml --backend mock --allow-dirty
```

The command writes a run directory under `results/runs/`. Then score and report it:

```bash
cosq evaluate results/runs/<RUN_DIR>
cosq analyze results/runs/<RUN_DIR>
cosq report results/runs/<RUN_DIR>
```

## Using OpenAI Models

Set your API key:

```bash
set OPENAI_API_KEY=sk-...        # Windows cmd
$env:OPENAI_API_KEY="sk-..."     # PowerShell
export OPENAI_API_KEY=sk-...     # macOS/Linux
```

Probe the model:

```bash
cosq probe --model configs/model/openai_gpt4o_mini.yaml
```

Run a small open-ended TruthfulQA pilot:

```bash
cosq run --config configs/experiment/pilot_truthfulqa_open_openai.yaml --allow-dirty
```

You can copy `configs/model/openai_template.yaml` and change `id` to another model available in your OpenAI account.

## Using Hugging Face Models

Install optional dependencies:

```bash
pip install -e ".[hf]"
```

For gated models, set `HF_TOKEN` and accept the model license on Hugging Face. Then copy one of the examples in `configs/model/`, replace `revision` with an immutable Hugging Face commit SHA, and run:

```bash
cosq probe --model configs/model/hf_llama3_8b_instruct.yaml
cosq run --config configs/experiment/pilot_truthfulqa_open.yaml --allow-dirty
```

Pinned revisions are required for local Hugging Face models so that a run refers to stable weights.

## Experiment Configuration

```yaml
name: smoke_mock
model: configs/model/mock.yaml
data:
  name: jsonl
  path: tests/fixtures/mini.jsonl
  n: 3
  seed: 1002
strategies:
  - name: direct
  - name: cot
  - name: cot_abstain
  - name: cosq
  - name: cosq_graded_gate
    label: cosq_graded_gate_mean060
    threshold: 0.60
    aggregator: mean
    on_empty: abstain
repeats: 1
open_ended: false
```

## Built-in Strategies

- `direct`: answer directly.
- `cot`: reason step by step, then answer.
- `cot_abstain`: CoT with permission to answer `I don't know`.
- `cosq`: binary Chain-of-Self-Questioning with a strict conjunctive gate.
- `cosq_gate`: CoSQ gate followed by CoT-style answer generation.
- `cosq_graded`: item-level 0-100 confidence scores with thresholding.
- `cosq_graded_gate`: graded gate followed by CoT-style answer generation.

## Data

CoSQ ships with a tiny JSONL fixture for tests. Benchmark datasets are loaded at runtime or supplied as JSONL files. Local dataset payloads are ignored by git to keep the package lightweight.

A custom JSONL dataset should contain:

```json
{"id":"q1","question":"What is the capital of France?","options":["Paris","Lyon"],"gold_index":0}
```

## Caching and Re-scoring

Runs can use a SQLite cache:

```bash
cosq run --config configs/experiment/pilot_truthfulqa_open_openai.yaml --cache results/cache.sqlite
```

Raw model outputs are written before scoring. You can improve parsers or metrics and re-run `evaluate`, `analyze`, or `report` without making new model calls.

## Development and PyPI Release

```bash
pip install -e ".[dev]"
pytest
ruff check .
python -m build
python -m twine check dist/*
```

To publish to PyPI:

```bash
python -m build
python -m twine upload dist/*
```

## Repository Layout

```text
src/cosq/
  backends/       # mock, local Hugging Face, OpenAI API
  strategies/     # direct, CoT, CoSQ, graded CoSQ variants
  decision/       # binary and confidence-based gates
  data/           # dataset adapters
  eval/           # offline scoring and metrics
  report/         # tables and statistical summaries
  prompts/        # versioned prompt templates
configs/
  experiment/     # runnable experiment YAML files
  model/          # model backend examples
tests/            # no-network test suite
```

## Citation

If you use CoSQ in academic work, please cite the accompanying paper once available. A BibTeX entry will be added after publication.

## License

MIT License.

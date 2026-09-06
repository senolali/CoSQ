"""The experiment loop.

Writes every completion to ``records.jsonl`` as it happens, before any parsing or
scoring, so an interrupted run loses nothing and a finished run can be re-scored
offline forever.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cosq import prompts
from cosq.config import ExperimentConfig, config_hash
from cosq.data.base import sample_questions
from cosq.runner.cache import CachingBackend, CompletionCache
from cosq.runner.manifest import build_manifest, git_state
from cosq.types import AnswerRecord, Question

ProgressFn = Callable[[int, int, str], None]


@dataclass(slots=True)
class RunResult:
    run_id: str
    run_dir: Path
    n_records: int
    questions: list[Question]
    records: list[AnswerRecord]
    cache_hits: int
    cache_misses: int
    total_cost_usd: float = 0.0


def make_run_id(config: ExperimentConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    model = config.model.id.rsplit("/", 1)[-1]
    return f"{stamp}__{config.name}__{model}__{config_hash(config)}"


def seed_everything(seed: int) -> None:
    """Seed every source of randomness the run can touch.

    No unseeded ``random`` or ``np.random`` calls anywhere in the package — that is
    what makes the dataset sample and any bootstrap reproduce exactly.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:  # pragma: no cover - only when torch is installed
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _record_to_json(record: AnswerRecord) -> str:
    return json.dumps(
        {
            "question_id": record.question_id,
            "strategy": record.strategy,
            "repeat": record.repeat,
            "answer_text": record.answer_text,
            "decision": record.decision,
            "needs": record.needs,
            "certainties": record.certainties,
            "n_needs": record.n_needs,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "latency_s": record.latency_s,
            "cost_usd": record.cost_usd,
            "meta": record.meta,
            "trace": [
                {"stage": turn.stage, "prompt": turn.prompt, "completion": turn.completion}
                for turn in record.trace
            ],
        },
        ensure_ascii=False,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    results_root: str | Path = "results/runs",
    cache_path: str | Path | None = "results/cache.sqlite",
    backend: Any = None,
    allow_dirty: bool = False,
    progress: ProgressFn | None = None,
) -> RunResult:
    """Execute the full grid and persist everything.

    ``allow_dirty`` defaults to False: a run started from a modified working tree
    cannot be mapped back to a commit, so it is refused rather than silently
    producing unattributable results.
    """
    state = git_state()
    if state.get("dirty") and not allow_dirty:
        raise RuntimeError(
            "the working tree has uncommitted changes, so this run could not be mapped "
            "back to a commit.\n"
            "  Check what changed:  git status --short\n"
            "  Then either commit it, or add --allow-dirty for a scratch run whose "
            "provenance you do not need."
        )

    seed_everything(config.seed)
    dataset = config.data.build()
    all_questions = dataset.load()
    questions, ids_sha256 = sample_questions(all_questions, config.data.n, config.data.seed)

    run_id = make_run_id(config)
    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_backend = backend if backend is not None else config.model.build()
    params = config.model.params()

    (run_dir / "config.resolved.yaml").write_text(
        json.dumps(config.to_dict(), indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = build_manifest(
        run_id=run_id,
        config=config.to_dict(),
        config_hash=config_hash(config),
        seed=config.seed,
        dataset={
            "name": config.data.name,
            "n": len(questions),
            "sample_seed": config.data.seed,
            "ids_sha256": ids_sha256,
        },
        backend=raw_backend.fingerprint(),
        prompts_digest=prompts.prompt_digest(config.lang),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    total = len(questions) * len(config.strategies) * config.repeats
    records: list[AnswerRecord] = []
    done = 0

    with CompletionCache(cache_path) as cache:
        with (run_dir / "records.jsonl").open("w", encoding="utf-8") as sink:
            for repeat in range(config.repeats):
                caching = CachingBackend(raw_backend, cache, repeat=repeat)
                for strategy_cfg in config.strategies:
                    strategy = strategy_cfg.build(caching, open_ended=config.open_ended)
                    for question in questions:
                        record = strategy.answer(question, params, repeat=repeat)
                        # The class name is not the condition name: one experiment may
                        # run the same strategy under several settings.
                        record.strategy = strategy_cfg.condition
                        records.append(record)
                        sink.write(_record_to_json(record) + "\n")
                        sink.flush()
                        done += 1
                        if progress is not None:
                            progress(done, total, f"{strategy_cfg.condition}/{question.id}")

        hits, misses = cache.hits, cache.misses

    total_cost = sum(record.cost_usd for record in records)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["cache"] = {"hits": hits, "misses": misses}
    # Hosted inference is metered, so what a run cost is part of its provenance.
    manifest["cost"] = {
        "total_usd": total_cost,
        "prompt_tokens": sum(record.prompt_tokens for record in records),
        "completion_tokens": sum(record.completion_tokens for record in records),
        "api_calls": sum(len(record.trace) for record in records),
    }
    manifest["backend_final"] = raw_backend.fingerprint()
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    return RunResult(
        run_id=run_id,
        run_dir=run_dir,
        n_records=len(records),
        questions=questions,
        records=records,
        cache_hits=hits,
        cache_misses=misses,
        total_cost_usd=total_cost,
    )


def load_records(run_dir: str | Path) -> list[AnswerRecord]:
    """Read back ``records.jsonl`` for offline scoring."""
    from cosq.types import Turn

    path = Path(run_dir) / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no records.jsonl in {run_dir}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(
                AnswerRecord(
                    question_id=row["question_id"],
                    strategy=row["strategy"],
                    repeat=row["repeat"],
                    answer_text=row["answer_text"],
                    decision=row["decision"],
                    trace=[Turn(**turn) for turn in row.get("trace", [])],
                    needs=row.get("needs", []),
                    certainties=row.get("certainties", []),
                    prompt_tokens=row.get("prompt_tokens", 0),
                    completion_tokens=row.get("completion_tokens", 0),
                    latency_s=row.get("latency_s", 0.0),
                    cost_usd=row.get("cost_usd", 0.0),
                    meta=row.get("meta", {}),
                )
            )
    return records

"""The ``cosq`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cosq import __version__
from cosq.config import ExperimentConfig
from cosq.dotenv import load_dotenv
from cosq.registry import available


def _progress(done: int, total: int, label: str) -> None:
    print(f"\r  [{done:>5}/{total}] {label:<48}", end="", file=sys.stderr, flush=True)


def _load_questions(config: ExperimentConfig) -> dict[str, Any]:
    from cosq.data.base import sample_questions

    questions = config.data.build().load()
    sample, _ = sample_questions(questions, config.data.n, config.data.seed)
    return {question.id: question for question in sample}


def cmd_ask(args: argparse.Namespace) -> int:
    from cosq.backends import load_backend
    from cosq.decision import ThresholdRule
    from cosq.registry import resolve
    from cosq.types import GenerationParams, Question

    backend = load_backend(args.model)
    strategy_cls = resolve("strategy", args.strategy)
    strategy = (
        strategy_cls(backend, rule=ThresholdRule(tau=args.tau))
        if args.strategy == "cosq"
        else strategy_cls(backend)
    )
    options = tuple(args.option) if args.option else ("yes", "no")
    question = Question(id="adhoc", text=args.question, options=options, gold_index=0)
    record = strategy.answer(question, GenerationParams())

    if args.show_trace:
        for turn in record.trace:
            print(f"\n--- {turn.stage} ---")
            print(turn.prompt.rstrip())
            print(f">>> {turn.completion.strip()}")
        print()
        if record.needs:
            print("Sub-facts and certainty:")
            for need, certainty in zip(record.needs, record.certainties, strict=True):
                print(f"  [{certainty:>9}] {need}")
            print(f"Decision: {record.decision.upper()} ({record.meta.get('decision_reason', '')})")
    print(f"\nAnswer: {record.answer_text.strip()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from cosq.runner import run_experiment

    config = ExperimentConfig.load(args.config)
    backend = None
    if args.backend:
        from cosq.backends import load_backend

        backend = load_backend(args.backend)
    print(
        f"{config.name}: {config.data.n} questions x {len(config.strategies)} conditions "
        f"x {config.repeats} repeats = {config.total_queries()} queries "
        f"(n = {config.data.n} questions is the statistical unit)",
        file=sys.stderr,
    )
    result = run_experiment(
        config,
        results_root=args.results_root,
        cache_path=None if args.no_cache else args.cache,
        backend=backend,
        allow_dirty=args.allow_dirty,
        max_workers=args.workers,
        progress=None if args.quiet else _progress,
    )
    print(file=sys.stderr)
    print(
        f"wrote {result.n_records} records to {result.run_dir} "
        f"(cache: {result.cache_hits} hits, {result.cache_misses} misses)",
        file=sys.stderr,
    )
    print(result.run_dir)
    return 0


def _score(run_dir: Path, config_path: str | None) -> tuple[list[Any], ExperimentConfig]:
    from cosq.eval import score_records
    from cosq.runner.runner import load_records

    # The run's own config.resolved.yaml is JSON, which YAML parses natively, so one
    # loader handles both it and a hand-written config.
    resolved = config_path or str(run_dir / "config.resolved.yaml")
    config = ExperimentConfig.load(resolved)
    records = load_records(run_dir)
    return score_records(records, _load_questions(config)), config


def cmd_evaluate(args: argparse.Namespace) -> int:
    from cosq.eval import compute_metrics
    from cosq.report import metrics_note, metrics_table

    run_dir = Path(args.run_dir)
    scored, _ = _score(run_dir, args.config)

    by_strategy: dict[str, list[Any]] = {}
    for record in scored:
        by_strategy.setdefault(record.strategy, []).append(record)
    metrics = {
        name: compute_metrics(items).to_dict() for name, items in sorted(by_strategy.items())
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics_table(metrics))
    print(f"\n{metrics_note()}")
    print(f"\nwrote {run_dir / 'metrics.json'}", file=sys.stderr)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from cosq.report import analyze_run

    run_dir = Path(args.run_dir)
    scored, _ = _score(run_dir, args.config)
    analysis = analyze_run(scored)
    (run_dir / "stats.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2, default=str))
    print(f"\nwrote {run_dir / 'stats.json'}", file=sys.stderr)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from cosq.eval import compute_metrics
    from cosq.report import analyze_run, metrics_note, metrics_table

    run_dir = Path(args.run_dir)
    scored, config = _score(run_dir, args.config)
    by_strategy: dict[str, list[Any]] = {}
    for record in scored:
        by_strategy.setdefault(record.strategy, []).append(record)
    metrics = {
        name: compute_metrics(items).to_dict() for name, items in sorted(by_strategy.items())
    }
    analysis = analyze_run(scored)

    out_dir = Path(args.out) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    body = [
        f"# {config.name}",
        "",
        f"Run: `{run_dir.name}`",
        "",
        "> Generated by `cosq report`. Do not edit by hand.",
        "",
        "## Metrics",
        "",
        metrics_table(metrics),
        "",
        metrics_note(),
        "",
        "## Analysis",
        "",
        "```json",
        json.dumps(analysis, indent=2, default=str),
        "```",
        "",
    ]
    target = out_dir / f"{run_dir.name}.md"
    target.write_text("\n".join(body), encoding="utf-8")
    print(target)
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Verify connectivity and a model name with one cheap call.

    Hosted model names are not guessable, and a wrong `provider` can route to a
    different model without erroring. Probe before committing a campaign to a config.
    """
    from cosq.backends import load_backend
    from cosq.types import GenerationParams

    backend = load_backend(args.model)
    params = GenerationParams(max_new_tokens=16)
    prompt = args.prompt
    print(
        f"probing {backend.fingerprint().get('model_id')} via {type(backend).__name__} ...",
        file=sys.stderr,
    )
    try:
        completion = backend.generate(prompt, params)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"  reply      : {completion.text.strip()[:200]!r}")
    print(f"  tokens     : {completion.prompt_tokens} in / {completion.completion_tokens} out")
    print(f"  cost       : ${completion.cost_usd:.6f}")
    print(f"  latency    : {completion.latency_s:.2f}s")
    fingerprint = backend.fingerprint()
    if fingerprint.get("pinned") == "false":
        print(
            "  WARNING    : this model is not version-pinned; the provider may change "
            "the weights behind this name without notice [M15]"
        )
    if completion.completion_tokens > 32:
        print(
            "  NOTE       : the reply exceeded the requested 16-token cap, so the "
            "token limit is probably NOT enforced by this endpoint [M16]"
        )
    # A base (non-instruction-tuned) model rambles instead of obeying a one-word
    # instruction, and would not follow the CoSQ stage prompts at all. A platform
    # model name such as "llama3-8b" does not say which variant is served.
    if len(completion.text.split()) > 12:
        print(
            "  WARNING    : the reply is long for a one-word instruction. Check that "
            "this is the instruction-tuned variant, not a base model"
        )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for kind in ("backend", "strategy", "decision", "dataset"):
        print(f"{kind:>9}: {', '.join(available(kind)) or '(none)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cosq", description=__doc__)
    parser.add_argument("--version", action="version", version=f"cosq {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a single question")
    ask.add_argument("question")
    ask.add_argument("--strategy", default="cosq")
    ask.add_argument("--model", default="mock", help="model config path, or 'mock'")
    ask.add_argument("--option", action="append", help="repeatable answer option")
    ask.add_argument("--tau", type=float, default=0.0)
    ask.add_argument("--show-trace", action="store_true")
    ask.set_defaults(func=cmd_ask)

    run = sub.add_parser("run", help="execute an experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--backend", help="override the backend, e.g. 'mock'")
    run.add_argument("--results-root", default="results/runs")
    run.add_argument("--cache", default="results/cache.sqlite")
    run.add_argument("--workers", type=int, default=1, help="parallel question workers")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--allow-dirty", action="store_true", help="permit an uncommitted tree")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)

    for name, handler, help_text in (
        ("evaluate", cmd_evaluate, "score a run and write metrics.json"),
        ("analyze", cmd_analyze, "run the pre-specified analysis, write stats.json"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("run_dir")
        p.add_argument("--config", help="experiment config (defaults to the run's own)")
        p.set_defaults(func=handler)

    report = sub.add_parser("report", help="regenerate tables from a run")
    report.add_argument("run_dir")
    report.add_argument("--config")
    report.add_argument("--out", default="results")
    report.set_defaults(func=cmd_report)

    probe = sub.add_parser("probe", help="verify a model config with one cheap call")
    probe.add_argument("--model", required=True, help="model config path")
    probe.add_argument("--prompt", default="Reply with the single word: ok", help="probe prompt")
    probe.set_defaults(func=cmd_probe)

    sub.add_parser("list", help="list registered plugins").set_defaults(func=cmd_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Read .env before anything reads os.environ, so the documented "copy .env.example
    # and fill it in" workflow actually works. Real environment variables win.
    load_dotenv()
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

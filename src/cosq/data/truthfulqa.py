"""TruthfulQA MC1 adapter.

**MC1, scored generation-then-match.** The model generates an answer and the parser
maps it to an option. Standard TruthfulQA MC1/MC2 numbers come from comparing option
*log-likelihoods*, which is a different protocol — so scores produced here must not be
placed next to published figures [M9].

The loader prefers a local JSONL cache so runs never depend on network availability;
``datasets`` is imported lazily and only when no cache exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from cosq.data.base import DatasetAdapter
from cosq.data.jsonl import JsonlDataset
from cosq.registry import register
from cosq.types import Question

DEFAULT_CACHE = Path("data/truthfulqa_mc1.jsonl")
_DATASET_SPECS = (
    ("truthfulqa/truthful_qa", "multiple_choice"),
    ("truthful_qa", "multiple_choice"),
)


@register("dataset", "truthfulqa_mc1")
class TruthfulQAMC1(DatasetAdapter):
    """817 questions; each has one correct option among several plausible falsehoods."""

    name = "truthfulqa_mc1"

    def __init__(self, cache_path: str | Path = DEFAULT_CACHE, *, allow_download: bool = True):
        self.cache_path = Path(cache_path)
        self.allow_download = allow_download

    def load(self) -> list[Question]:
        if self.cache_path.is_file():
            return JsonlDataset(self.cache_path).load()
        if not self.allow_download:
            raise FileNotFoundError(
                f"no cached dataset at {self.cache_path} and downloads are disabled"
            )
        questions = self._download()
        self._write_cache(questions)
        return questions

    def _download(self) -> list[Question]:  # pragma: no cover - requires the network
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "downloading TruthfulQA needs 'datasets': pip install datasets. "
                f"Alternatively place a JSONL cache at {self.cache_path}."
            ) from exc

        last_error: Exception | None = None
        for path, name in _DATASET_SPECS:
            try:
                rows = load_dataset(path, name, split="validation")
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(
                "could not download TruthfulQA from Hugging Face. "
                f"Place a JSONL cache at {self.cache_path} or check datasets/hub compatibility."
            ) from last_error

        questions = []
        for index, row in enumerate(rows):
            targets = row["mc1_targets"]
            labels = list(targets["labels"])
            questions.append(
                Question(
                    id=f"tqa-mc1-{index:04d}",
                    text=row["question"],
                    options=tuple(targets["choices"]),
                    gold_index=labels.index(1),
                )
            )
        return questions

    def _write_cache(self, questions: list[Question]) -> None:  # pragma: no cover
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            for question in questions:
                handle.write(
                    json.dumps(
                        {
                            "id": question.id,
                            "question": question.text,
                            "options": list(question.options),
                            "gold_index": question.gold_index,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

"""MMLU multiple-choice adapter with a local JSONL cache."""

from __future__ import annotations

import json
from pathlib import Path

from cosq.data.base import DatasetAdapter
from cosq.data.jsonl import JsonlDataset
from cosq.registry import register
from cosq.types import Question

DEFAULT_CACHE = Path("data/mmlu_test.jsonl")


@register("dataset", "mmlu")
class MMLU(DatasetAdapter):
    name = "mmlu"

    def __init__(self, cache_path: str | Path = DEFAULT_CACHE, *, allow_download: bool = True):
        self.cache_path = Path(cache_path)
        self.allow_download = allow_download

    def load(self) -> list[Question]:
        if self.cache_path.is_file():
            return JsonlDataset(self.cache_path).load()
        if not self.allow_download:
            raise FileNotFoundError(f"no cached MMLU file at {self.cache_path}")
        questions = self._download()
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
                            "meta": question.meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return questions

    def _download(self) -> list[Question]:  # pragma: no cover
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "downloading MMLU needs 'datasets'; alternatively place a JSONL cache "
                "at data/mmlu_test.jsonl"
            ) from exc
        rows = load_dataset("cais/mmlu", "all", split="test")
        return [
            Question(
                id=f"mmlu-{index:05d}",
                text=str(row["question"]),
                options=tuple(str(x) for x in row["choices"]),
                gold_index=int(row["answer"]),
                meta={"subject": str(row.get("subject", "unknown"))},
            )
            for index, row in enumerate(rows)
        ]

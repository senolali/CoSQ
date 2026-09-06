"""Natural Questions short-answer adapter for ordinary factual QA.

Unlike TruthfulQA MC1, this dataset has no curated false options. We therefore use
short-answer scoring: a committed answer that does not match the reference answer is
``wrong`` rather than ``unparseable``. This makes HR interpretable for the second
benchmark while keeping the mode explicit in each question's metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cosq.data.base import DatasetAdapter
from cosq.data.jsonl import JsonlDataset
from cosq.registry import register
from cosq.types import Question

DEFAULT_CACHE = Path("data/natural_questions_slim_short_answer_validation.jsonl")
DATASET_NAME = "BOB12311/natural-questions-slim-short-answer"


@register("dataset", "nq_short")
class NaturalQuestionsShort(DatasetAdapter):
    """Natural Questions short-answer validation split, flattened to QA pairs."""

    name = "nq_short"

    def __init__(self, cache_path: str | Path = DEFAULT_CACHE, *, allow_download: bool = True):
        self.cache_path = Path(cache_path)
        self.allow_download = allow_download

    def load(self) -> list[Question]:
        if self.cache_path.is_file():
            return self._load_cache()
        if not self.allow_download:
            raise FileNotFoundError(
                f"no cached dataset at {self.cache_path} and downloads are disabled"
            )
        questions = self._download()
        self._write_cache(questions)
        return questions

    def _load_cache(self) -> list[Question]:
        questions = JsonlDataset(self.cache_path).load()
        return [
            Question(
                id=question.id,
                text=question.text,
                options=question.options,
                gold_index=question.gold_index,
                meta={**question.meta, "scoring": "short_answer", "all_options_correct": True},
            )
            for question in questions
        ]

    def _download(self) -> list[Question]:  # pragma: no cover - requires the network
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "downloading Natural Questions needs 'datasets': pip install datasets. "
                f"Alternatively place a JSONL cache at {self.cache_path}."
            ) from exc

        rows = load_dataset(DATASET_NAME, split="validation")
        questions = []
        for index, row in enumerate(rows):
            question_text = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            if not question_text or not answer:
                continue
            questions.append(self._question(index, question_text, answer))
        return questions

    @staticmethod
    def _question(index: int, question_text: str, answer: str) -> Question:
        return Question(
            id=f"nq-short-{index:05d}",
            text=question_text,
            options=(answer,),
            gold_index=0,
            meta={"scoring": "short_answer", "all_options_correct": True},
        )

    def _write_cache(self, questions: list[Question]) -> None:  # pragma: no cover
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            for question in questions:
                row: dict[str, Any] = {
                    "id": question.id,
                    "question": question.text,
                    "options": list(question.options),
                    "gold_index": question.gold_index,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

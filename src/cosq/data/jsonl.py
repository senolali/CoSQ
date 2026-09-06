"""A JSONL dataset adapter — the format fixtures and cached datasets use."""

from __future__ import annotations

import json
from pathlib import Path

from cosq.data.base import DatasetAdapter
from cosq.registry import register
from cosq.types import Question


@register("dataset", "jsonl")
class JsonlDataset(DatasetAdapter):
    """One JSON object per line: ``id``, ``question``, ``options``, ``gold_index``."""

    name = "jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Question]:
        if not self.path.is_file():
            raise FileNotFoundError(f"dataset file not found: {self.path}")
        questions = []
        with self.path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    questions.append(
                        Question(
                            id=str(row["id"]),
                            text=row["question"],
                            options=tuple(row["options"]),
                            gold_index=int(row["gold_index"]),
                        )
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"{self.path}:{lineno}: malformed row: {exc}") from exc
        if not questions:
            raise ValueError(f"{self.path} contains no questions")
        return questions

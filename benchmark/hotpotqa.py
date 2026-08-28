from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class EvidenceDocument:
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass(frozen=True)
class HotpotTask:
    task_id: str
    question: str
    question_type: str
    level: str
    documents: tuple[EvidenceDocument, ...]
    _gold_answer: str
    _gold_supporting_facts: tuple[tuple[str, int], ...]

    def workflow_view(self) -> dict[str, Any]:
        """Return the only task representation visible before evaluation."""
        return {
            "task_id": self.task_id,
            "question": self.question,
            "question_type": self.question_type,
            "level": self.level,
            "documents": [
                {"title": document.title, "sentences": list(document.sentences)}
                for document in self.documents
            ],
        }

    def evaluation_gold(self) -> dict[str, Any]:
        return {
            "answer": self._gold_answer,
            "supporting_facts": [list(fact) for fact in self._gold_supporting_facts],
        }


class HotpotQAAdapter:
    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self._tasks: dict[str, HotpotTask] | None = None

    def _load(self) -> dict[str, HotpotTask]:
        if self._tasks is not None:
            return self._tasks
        rows = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        tasks: dict[str, HotpotTask] = {}
        for row in rows:
            task = HotpotTask(
                task_id=str(row["_id"]),
                question=str(row["question"]),
                question_type=str(row["type"]),
                level=str(row.get("level", "")),
                documents=tuple(
                    EvidenceDocument(str(title), tuple(map(str, sentences)))
                    for title, sentences in row["context"]
                ),
                _gold_answer=str(row.get("answer", "")),
                _gold_supporting_facts=tuple(
                    (str(title), int(index)) for title, index in row.get("supporting_facts", [])
                ),
            )
            if task.task_id in tasks:
                raise ValueError(f"duplicate HotpotQA task id: {task.task_id}")
            tasks[task.task_id] = task
        self._tasks = tasks
        return tasks

    def get(self, task_id: str) -> HotpotTask:
        return self._load()[task_id]

    def deterministic_pilot_ids(self, seed: str = "fllm-2026-pilot60-v1") -> dict[str, str]:
        selected: dict[str, str] = {}
        for question_type in ("bridge", "comparison"):
            eligible = [
                task
                for task in self._load().values()
                if task.question_type == question_type and len(task.documents) >= 2
            ]
            if not eligible:
                raise ValueError(f"no eligible {question_type} HotpotQA task")
            chosen = min(
                eligible,
                key=lambda task: (
                    hashlib.sha256(f"{seed}:{task.task_id}".encode()).hexdigest(),
                    task.task_id,
                ),
            )
            selected[question_type] = chosen.task_id
        return selected

    def dataset_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.dataset_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class DeterministicRetriever:
    """Small deterministic BM25 retriever over a task's supplied paragraphs."""

    def __init__(self, documents: Iterable[EvidenceDocument]) -> None:
        self.documents = tuple(documents)
        self.term_frequencies = [Counter(_tokens(document.text)) for document in self.documents]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.document_frequency = Counter(
            token for frequencies in self.term_frequencies for token in frequencies
        )

    def ranked(self, query: str) -> tuple[EvidenceDocument, ...]:
        query_terms = _tokens(query)
        scored = []
        count = len(self.documents)
        for index, (document, frequencies) in enumerate(
            zip(self.documents, self.term_frequencies, strict=True)
        ):
            score = 0.0
            length = self.lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                inverse = math.log(1 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / max(1, self.average_length))
                score += inverse * frequency * 2.5 / denominator
            scored.append((-score, document.title, index, document))
        scored.sort(key=lambda row: row[:3])
        return tuple(row[3] for row in scored)

    def next_unseen(self, query: str, seen_titles: set[str]) -> EvidenceDocument | None:
        return next(
            (document for document in self.ranked(query) if document.title not in seen_titles),
            None,
        )

    def next_unseen_many(
        self, query: str, seen_titles: set[str], *, limit: int = 2
    ) -> tuple[EvidenceDocument, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return tuple(
            document
            for document in self.ranked(query)
            if document.title not in seen_titles
        )[:limit]

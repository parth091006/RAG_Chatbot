from __future__ import annotations


class EvaluationDataset:
    """A minimal dataset structure for QA and retrieval evaluation."""

    def __init__(self, rows: list[dict[str, str]] | None = None):
        self.rows = rows or []

    def add(self, question: str, expected_answer: str, expected_sources: list[str] | None = None) -> None:
        self.rows.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": expected_sources or [],
            }
        )

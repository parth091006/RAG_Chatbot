from __future__ import annotations

from typing import Any


class EvaluationDataset:
    """Dataset structure for retrieval and generation evaluation."""

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
    ):
        self.rows = rows or []

    def add(
        self,
        question: str,
        expected_answer: str = "",
        expected_sources: list[str] | None = None,
        relevant_chunks: list[str] | None = None,
    ) -> None:
        self.rows.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": expected_sources or [],
                "relevant_chunks": relevant_chunks or [],
            }
        )

    def add_retrieval_example(
        self,
        question: str,
        relevant_chunks: list[str],
    ) -> None:
        """Add an example specifically for retrieval evaluation."""

        self.rows.append(
            {
                "question": question,
                "expected_answer": "",
                "expected_sources": [],
                "relevant_chunks": relevant_chunks,
            }
        )

    def add_generation_example(
        self,
        question: str,
        expected_answer: str,
        expected_sources: list[str] | None = None,
    ) -> None:
        """Add an example specifically for generation evaluation."""

        self.rows.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": expected_sources or [],
                "relevant_chunks": [],
            }
        )
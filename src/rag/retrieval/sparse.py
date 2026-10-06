from __future__ import annotations

import math
import re
from collections import Counter


class BM25Retriever:
    """Lightweight BM25 retriever for sparse lexical search."""

    def __init__(
        self,
        documents: list[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents or []
        self.k1 = k1
        self.b = b

        self.document_tokens: list[list[str]] = []
        self.document_lengths: list[int] = []
        self.term_document_frequency: Counter[str] = Counter()
        self.average_document_length = 0.0

        self._fit()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _fit(self) -> None:
        self.document_tokens = [
            self._tokenize(document)
            for document in self.documents
        ]

        self.document_lengths = [
            len(tokens) for tokens in self.document_tokens
        ]

        if not self.document_lengths:
            self.average_document_length = 0.0
            return

        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
        )

        self.term_document_frequency = Counter()

        for tokens in self.document_tokens:
            unique_terms = set(tokens)

            for term in unique_terms:
                self.term_document_frequency[term] += 1

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        if not self.documents or top_k <= 0:
            return []

        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        document_count = len(self.documents)
        query_term_counts = Counter(query_terms)

        scores: list[tuple[int, float]] = []

        for index, tokens in enumerate(self.document_tokens):
            document_length = self.document_lengths[index]

            if document_length == 0:
                scores.append((index, 0.0))
                continue

            term_counts = Counter(tokens)
            score = 0.0

            for term, query_frequency in query_term_counts.items():
                term_frequency = term_counts.get(term, 0)

                if term_frequency == 0:
                    continue

                document_frequency = self.term_document_frequency.get(term, 0)

                # Standard BM25 IDF with smoothing.
                idf = math.log(
                    1.0
                    + (
                        document_count
                        - document_frequency
                        + 0.5
                    )
                    / (
                        document_frequency + 0.5
                    )
                )

                denominator = (
                    term_frequency
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * (
                            document_length
                            / self.average_document_length
                        )
                    )
                )

                term_score = (
                    idf
                    * (
                        (
                            term_frequency
                            * (self.k1 + 1.0)
                        )
                        / denominator
                    )
                )

                score += query_frequency * term_score

            scores.append((index, score))

        scores.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scores[:top_k]
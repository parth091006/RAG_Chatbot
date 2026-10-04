from __future__ import annotations

from collections import Counter


class BM25Retriever:
    """A lightweight BM25-style retriever for keyword matching."""

    def __init__(self, documents: list[str] | None = None):
        self.documents = documents or []

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        if not self.documents:
            return []

        query_terms = query.lower().split()
        doc_scores: list[tuple[int, float]] = []
        for index, document in enumerate(self.documents):
            terms = document.lower().split()
            counter = Counter(terms)
            score = 0.0
            for term in query_terms:
                if term in counter:
                    score += counter[term] * 1.5
            doc_scores.append((index, score))

        doc_scores.sort(key=lambda item: item[1], reverse=True)
        return doc_scores[:top_k]

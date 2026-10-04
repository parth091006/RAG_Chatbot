from __future__ import annotations


class Reranker:
    """A second-stage reranker that keeps the best candidates."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def rerank(self, candidates: list[tuple[int, float]]) -> list[tuple[int, float]]:
        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        return ranked[: self.top_k]

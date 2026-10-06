from __future__ import annotations

from .dense import DenseRetriever
from .sparse import BM25Retriever


class HybridRetriever:
    """Hybrid dense + sparse retriever using Reciprocal Rank Fusion."""

    def __init__(
        self,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        rrf_k: int = 60,
    ):
        self.documents = documents or []
        self.embeddings = embeddings or []
        self.rrf_k = rrf_k

        self.dense = DenseRetriever(self.embeddings)
        self.sparse = BM25Retriever(self.documents)

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []

        dense_hits = self.dense.search(
            query_embedding,
            top_k=top_k,
        )

        sparse_hits = self.sparse.search(
            query,
            top_k=top_k,
        )

        fused_scores: dict[int, float] = {}

        # Reciprocal Rank Fusion:
        #
        # score = 1 / (rrf_k + rank)
        #
        # This combines rankings instead of directly adding
        # incompatible dense and sparse score scales.

        for rank, (index, _) in enumerate(dense_hits, start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        for rank, (index, _) in enumerate(sparse_hits, start=1):
            fused_scores[index] = fused_scores.get(index, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked[:top_k]
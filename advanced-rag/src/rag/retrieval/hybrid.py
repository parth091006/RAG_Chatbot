from __future__ import annotations

from .dense import DenseRetriever
from .sparse import BM25Retriever


class HybridRetriever:
    def __init__(self, documents: list[str] | None = None, embeddings: list[list[float]] | None = None):
        self.documents = documents or []
        self.embeddings = embeddings or []
        self.dense = DenseRetriever(self.embeddings)
        self.sparse = BM25Retriever(self.documents)

    def search(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        dense_hits = self.dense.search(query_embedding, top_k=top_k)
        sparse_hits = self.sparse.search(query, top_k=top_k)
        fused: dict[int, float] = {}

        for idx, score in dense_hits:
            fused[idx] = fused.get(idx, 0.0) + score
        for idx, score in sparse_hits:
            fused[idx] = fused.get(idx, 0.0) + score

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

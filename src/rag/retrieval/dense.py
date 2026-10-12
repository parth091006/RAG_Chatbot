from __future__ import annotations

import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseRetriever:
    def __init__(self, embeddings: list[list[float]] | None = None):
        self.embeddings = embeddings or []
        self._candidate_norms = [
            math.sqrt(sum(b * b for b in candidate))
            for candidate in self.embeddings
        ]

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        if not self.embeddings or not query_embedding or top_k <= 0:
            return []

        norm_q = math.sqrt(sum(a * a for a in query_embedding))
        if norm_q == 0.0:
            return [(index, 0.0) for index in range(min(top_k, len(self.embeddings)))]

        ranked: list[tuple[int, float]] = []
        for index, candidate in enumerate(self.embeddings):
            norm_c = self._candidate_norms[index]
            if norm_c == 0.0:
                score = 0.0
            else:
                dot = sum(a * b for a, b in zip(query_embedding, candidate))
                score = dot / (norm_q * norm_c)
            ranked.append((index, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

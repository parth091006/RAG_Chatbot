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

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        ranked: list[tuple[int, float]] = []
        for index, candidate in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, candidate)
            ranked.append((index, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

from __future__ import annotations


def recall_at_k(relevant: set[int], retrieved: list[int], k: int | None = None) -> float:
    limit = k if k is not None else len(retrieved)
    top = retrieved[:limit]
    hits = sum(1 for idx in top if idx in relevant)
    return hits / max(len(relevant), 1)


def mrr(relevant: set[int], retrieved: list[int]) -> float:
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1 / rank
    return 0.0

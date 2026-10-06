from __future__ import annotations

from src.rag.evaluation.retrieval_metrics import mrr, recall_at_k


def test_chunk_id_metrics() -> None:
    relevant = {"doc1_chunk_07", "doc1_chunk_12"}
    retrieved = ["doc1_chunk_03", "doc1_chunk_07", "doc1_chunk_12"]

    assert recall_at_k(relevant, retrieved, 1) == 0.0
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert mrr(relevant, retrieved) == 0.5
from __future__ import annotations

from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.sparse import BM25Retriever


def test_dense_retriever_returns_scores() -> None:
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    retriever = DenseRetriever(embeddings)
    ranked = retriever.search([1.0, 0.0], top_k=1)
    assert ranked
    assert ranked[0][0] in {0, 1}


def test_sparse_retriever_returns_scores() -> None:
    docs = ["transformer attention mechanism", "vision model architecture"]
    retriever = BM25Retriever(docs)
    ranked = retriever.search("attention", top_k=1)
    assert ranked
    assert ranked[0][0] in {0, 1}

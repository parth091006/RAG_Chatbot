from __future__ import annotations

import pytest

from src.rag.embeddings.embedder import Embedder


def test_bow_embedding() -> None:
    embedder = Embedder("bow")

    embeddings = embedder.embed_many(
        [
            "self attention",
            "transformer attention",
        ]
    )

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == len(embeddings[1])


def test_tfidf_embedding() -> None:
    embedder = Embedder("tfidf")

    embeddings = embedder.embed_many(
        [
            "self attention",
            "transformer attention",
        ]
    )

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == len(embeddings[1])


def test_invalid_embedding_method() -> None:
    with pytest.raises(ValueError):
        Embedder("invalid")


def test_semantic_embedding() -> None:
    embedder = Embedder("semantic")

    embedding = embedder.embed("What is self-attention?")

    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)
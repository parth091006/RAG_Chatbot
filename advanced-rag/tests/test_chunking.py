from __future__ import annotations

from src.rag.chunking.chunker import chunk_document


def test_chunk_document_returns_multiple_chunks() -> None:
    text = " ".join([f"word_{i}" for i in range(1000)])
    chunks = chunk_document(text, chunk_size=20, overlap=5)
    assert len(chunks) > 0
    assert all(len(chunk.strip()) > 0 for chunk in chunks)

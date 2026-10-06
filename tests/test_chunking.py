from __future__ import annotations

from src.rag.chunking.chunker import chunk_document, section_aware_chunks


def test_chunk_document_returns_multiple_chunks() -> None:
    text = " ".join([f"word_{i}" for i in range(1000)])
    chunks = chunk_document(text, chunk_size=20, overlap=5)
    assert len(chunks) > 0
    assert all(len(chunk.strip()) > 0 for chunk in chunks)


def test_section_aware_chunks_preserve_headings() -> None:
    pages = [
        "2.1 The key-query-value self-attention mechanism Query and key vectors determine attention weights.",
        "3.1 Multi-head Self-Attention Multiple heads capture different relationships.",
    ]

    chunks = section_aware_chunks(
        pages,
        document_id="doc001",
        document_name="paper.pdf",
        chunk_size=50,
        overlap=5,
    )

    assert len(chunks) == 2
    assert chunks[0]["section"] == "2.1 The key-query-value self-attention mechanism"
    assert chunks[1]["section"] == "3.1 Multi-head Self-Attention"

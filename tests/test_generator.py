from __future__ import annotations

from src.rag.generation.generator import Generator


def test_generator_returns_answer_and_sources_without_prompt() -> None:
    contexts = [
        {
            "chunk_id": "doc001_chunk_001",
            "document_name": "paper.pdf",
            "page_number": 2,
            "text": "Self-attention computes weighted combinations of value vectors using queries and keys.",
        }
    ]

    result = Generator().generate("What is self-attention?", contexts)

    assert set(result) == {"answer", "sources"}
    assert "Self-attention computes weighted combinations" in result["answer"]
    assert result["sources"] == [
        {
            "citation_id": "[doc001_chunk_001]",
            "document": "paper.pdf",
            "page": 2,
            "chunk_id": "doc001_chunk_001",
        }
    ]
    assert "Context:" not in result["answer"]
    assert "Requirements:" not in result["answer"]
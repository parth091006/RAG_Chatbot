from __future__ import annotations

from pathlib import Path

from src.rag.ingestion.parser import extract_text_from_pdf
from src.rag.pipeline import PhaseOneRAG


def test_phase_one_pipeline_with_real_pdf() -> None:
    pdf_path = Path("data/raw/self-attention-transformers-2023.pdf")

    text = extract_text_from_pdf(pdf_path)
    assert len(text) > 500
    assert "self" in text.lower()

    rag = PhaseOneRAG(pdf_path)
    chunks = rag.pdf_to_chunks()
    assert len(chunks) >= 2
    first_chunk = chunks[0]
    assert first_chunk["chunk_id"].startswith("doc_self-attention-transformers-2023_chunk_")
    assert first_chunk["document_name"] == pdf_path.name
    assert first_chunk["page_number"] >= 1
    assert first_chunk["text"]

    embeddings = rag.chunks_to_embeddings(chunks)
    assert len(embeddings) == len(chunks)
    assert chunks[0]["embedding"] == embeddings[0]

    question = "What is self-attention?"
    results = rag.question_to_top_k_chunks(question, top_k=5)
    assert len(results) > 0
    assert all(chunk["text"] for chunk in results)
    scored = rag.question_to_scored_chunks(question, top_k=5)
    assert scored[0]["chunk"]["chunk_id"] == results[0]["chunk_id"]
    assert isinstance(scored[0]["score"], float)

    answer = rag.answer_question(question)
    assert len(answer["answer"]) > 20
    assert "self" in answer["answer"].lower() or "attention" in answer["answer"].lower()
    assert answer["sources"]
    assert "chunk_id" in answer["sources"][0]
    assert "page" in answer["sources"][0]
    assert "Context" not in answer["answer"]
    assert "Requirements:" not in answer["answer"]

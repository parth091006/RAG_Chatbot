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

    embeddings = rag.chunks_to_embeddings(chunks)
    assert len(embeddings) == len(chunks)

    question = "What is self-attention?"
    results = rag.question_to_top_k_chunks(question, top_k=5)
    assert len(results) > 0
    assert all(chunk for chunk in results)

    answer = rag.answer_question(question)
    assert len(answer) > 20
    assert "self" in answer.lower() or "attention" in answer.lower()

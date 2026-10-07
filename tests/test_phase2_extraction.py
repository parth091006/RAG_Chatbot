import json
from pathlib import Path

import pytest

from src.rag.chunking.chunker import section_aware_chunks
from src.rag.pipeline import PhaseOneRAG


PDF_PATH = Path("data/raw/self-attention-transformers-2023.pdf")
SNAPSHOT_PATH = Path("tests/fixtures/legacy_chunks_snapshot.json")


def test_legacy_extraction_matches_chunk_snapshot() -> None:
    snapshot = json.loads(
        SNAPSHOT_PATH.read_text(encoding="utf-8")
    )
    pipeline = PhaseOneRAG(PDF_PATH)
    chunks = pipeline.pdf_to_chunks()

    assert pipeline.extraction == "legacy-pypdf"
    assert pipeline.chunk_count == snapshot["chunk_count"]
    assert pipeline.total_page_characters == snapshot["total_characters"]
    assert pipeline.total_chunk_characters == snapshot["total_chunk_characters"]

    for chunk, expected in zip(chunks, snapshot["chunks"]):
        assert chunk["chunk_id"] == expected["chunk_id"]
        assert chunk["document_id"] == expected["document_id"]
        assert chunk["document_name"] == expected["document_name"]
        assert chunk["page_number"] == expected["page_number"]
        assert len(str(chunk["text"]).split()) == expected["word_count"]
        assert chunk["chunk_text_hash"] == expected[
            "sha256_normalized_text"
        ][:12]


def test_current_extraction_matches_recorded_phase2_stats() -> None:
    pipeline = PhaseOneRAG(
        PDF_PATH,
        extraction="current",
    )
    chunks = pipeline.pdf_to_chunks()

    assert len(chunks) == 38
    assert pipeline.chunk_count == 38
    assert pipeline.total_page_characters == 40951
    assert pipeline.total_chunk_characters == sum(
        len(str(chunk["text"]))
        for chunk in chunks
    )


def test_primary_only_extraction_builds_and_exposes_stats() -> None:
    pipeline = PhaseOneRAG(
        PDF_PATH,
        extraction="primary-only",
    )
    chunks = pipeline.pdf_to_chunks()

    assert chunks
    assert pipeline.chunk_count == len(chunks)
    assert pipeline.total_page_characters > 0
    assert pipeline.total_chunk_characters > 0


def test_extraction_validation_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="Unsupported extraction"):
        PhaseOneRAG(
            PDF_PATH,
            extraction="unknown",
        )


def test_section_aware_chunks_do_not_receive_phase2_hashes() -> None:
    chunks = section_aware_chunks(
        [
            "2.1 The key-query-value self-attention mechanism "
            "Query and key vectors determine attention weights."
        ],
        document_id="doc001",
        document_name="paper.pdf",
        chunk_size=50,
        overlap=5,
    )

    assert chunks
    assert "chunk_text_hash" not in chunks[0]

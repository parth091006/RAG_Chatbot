from __future__ import annotations

import shutil
from pathlib import Path

from src.rag.ingestion.loader import (
    DocumentLoader,
    compute_file_sha256,
    load_pdf_paths,
)
from src.rag.ingestion.metadata import DocumentMetadata


def test_load_pdf_paths_returns_files(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = load_pdf_paths(tmp_path)

    assert pdf in result


def test_compute_file_sha256(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world", encoding="utf-8")

    digest = compute_file_sha256(test_file)
    # sha256 of "hello world"
    assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_document_metadata_dynamic_pdf_extraction() -> None:
    pdf_path = Path("data/raw/self-attention-transformers-2023.pdf")
    meta = DocumentMetadata.from_file(pdf_path)

    assert meta.document_id == "doc_self-attention-transformers-2023"
    assert meta.filename == "self-attention-transformers-2023.pdf"
    assert meta.pages == 18
    assert meta.file_size_bytes > 0
    assert len(meta.sha256_hash) == 64
    assert meta.title is not None
    assert "Self-Attention" in meta.title
    assert meta.creation_date is not None


def test_document_metadata_to_and_from_dict() -> None:
    pdf_path = Path("data/raw/self-attention-transformers-2023.pdf")
    original = DocumentMetadata.from_file(pdf_path, custom={"tag": "nlp"})

    data = original.to_dict()
    assert isinstance(data, dict)
    assert data["custom"] == {"tag": "nlp"}

    restored = DocumentMetadata.from_dict(data)
    assert restored.document_id == original.document_id
    assert restored.sha256_hash == original.sha256_hash
    assert restored.custom == {"tag": "nlp"}


def test_document_loader_deduplication(tmp_path: Path) -> None:
    source_pdf = Path("data/raw/self-attention-transformers-2023.pdf")
    pdf_a = tmp_path / "paper_a.pdf"
    pdf_b = tmp_path / "paper_b_copy.pdf"

    shutil.copyfile(source_pdf, pdf_a)
    shutil.copyfile(source_pdf, pdf_b)

    loader = DocumentLoader()

    # Ingest first file
    meta_a, is_dup_a = loader.load_file(pdf_a)
    assert not is_dup_a
    assert meta_a.document_id == "doc_paper_a"

    # Ingest duplicate file
    meta_b, is_dup_b = loader.load_file(pdf_b, deduplicate=True)
    assert is_dup_b
    assert meta_b.document_id == "doc_paper_a"  # points to original
    assert meta_b.sha256_hash == meta_a.sha256_hash

    # Check is_duplicate_file helper
    assert loader.is_duplicate_file(pdf_b)


def test_document_loader_directory_scan(tmp_path: Path) -> None:
    source_pdf = Path("data/raw/self-attention-transformers-2023.pdf")
    pdf_1 = tmp_path / "doc1.pdf"
    pdf_2 = tmp_path / "doc2.pdf"
    shutil.copyfile(source_pdf, pdf_1)
    shutil.copyfile(source_pdf, pdf_2)  # identical copy

    # Also create a different small PDF
    pdf_3 = tmp_path / "doc3.pdf"
    pdf_3.write_bytes(b"%PDF-1.4 unique content\n")

    loader = DocumentLoader()
    report = loader.load_directory(tmp_path, deduplicate=True)

    assert report.total_scanned == 3
    assert report.ingested_count == 2
    assert report.duplicate_count == 1
    assert report.duplicates[0]["sha256_hash"] == compute_file_sha256(pdf_1)
    assert len(loader.list_documents()) == 2

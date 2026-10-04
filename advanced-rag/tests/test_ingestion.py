from __future__ import annotations

from pathlib import Path

from src.rag.ingestion.loader import load_pdf_paths


def test_load_pdf_paths_returns_files(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = load_pdf_paths(tmp_path)

    assert pdf in result

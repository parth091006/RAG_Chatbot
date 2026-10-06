from __future__ import annotations

from src.rag.citations.citation_manager import CitationManager


def test_citation_manager_registers_and_resolves() -> None:
    manager = CitationManager()
    manager.register("[DOC1-P3-C2]", "attention.pdf", "3", "doc1_p3_c2")

    resolved = manager.resolve("[DOC1-P3-C2]")

    assert resolved["document"] == "attention.pdf"
    assert resolved["page"] == "3"

def test_citation_manager_returns_unknown_for_missing_citation() -> None:
    manager = CitationManager()

    resolved = manager.resolve("[UNKNOWN]")

    assert resolved == {
        "document": "unknown",
        "page": "unknown",
        "chunk_id": "unknown",
    }
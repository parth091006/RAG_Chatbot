from __future__ import annotations

from src.rag.citations.citation_manager import CitationManager


def test_citation_manager_registers_and_resolves() -> None:
    manager = CitationManager()
    manager.register("[DOC1-P3-C2]", "attention.pdf", "3", "doc1_p3_c2")

    resolved = manager.resolve("[DOC1-P3-C2]")

    assert resolved["document"] == "attention.pdf"
    assert resolved["page"] == "3"

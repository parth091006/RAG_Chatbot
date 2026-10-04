from __future__ import annotations

from typing import Iterable

from .strategies import fixed_size_chunks


def chunk_document(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    strategy: str = "fixed",
) -> list[str]:
    """Split a document text into chunks using a simple chunking strategy."""
    if not text:
        return []
    if strategy == "fixed":
        return fixed_size_chunks(text, chunk_size=chunk_size, overlap=overlap)
    return fixed_size_chunks(text, chunk_size=chunk_size, overlap=overlap)

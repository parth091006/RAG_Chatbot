from __future__ import annotations

import re
from typing import TypedDict

from .strategies import fixed_size_chunks


class Chunk(TypedDict, total=False):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    embedding: list[float]
    section: str


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


def chunk_pages(
    pages: list[str],
    document_id: str,
    document_name: str,
    chunk_size: int = 250,
    overlap: int = 50,
) -> list[Chunk]:
    """Create stable, page-aware chunk records from extracted PDF pages."""
    chunks: list[Chunk] = []
    chunk_number = 1
    for page_number, page_text in enumerate(pages, start=1):
        page_chunks = fixed_size_chunks(page_text, chunk_size=chunk_size, overlap=overlap)
        for text in page_chunks:
            chunks.append(
                {
                    "chunk_id": f"{document_id}_chunk_{chunk_number:03d}",
                    "document_id": document_id,
                    "document_name": document_name,
                    "page_number": page_number,
                    "text": text,
                }
            )
            chunk_number += 1
    return chunks


SECTION_HEADINGS = (
    "2.1 The key-query-value self-attention mechanism",
    "2.2 Position representations",
    "2.3 Elementwise nonlinearity",
    "2.4 Future masking",
    "3.1 Multi-head Self-Attention",
    "3.2 Layer Norm",
    "3.3 Residual Connections",
    "3.4 Attention logit scaling",
    "3.5 Transformer Encoder",
    "3.6 Transformer Decoder",
)


def section_aware_chunks(
    pages: list[str],
    document_id: str,
    document_name: str,
    chunk_size: int = 250,
    overlap: int = 50,
) -> list[Chunk]:
    """Split pages at known section headings, then chunk within each section."""
    normalized_headings = {re.sub(r"\s+", " ", heading).strip(): heading for heading in SECTION_HEADINGS}
    heading_pattern = re.compile(
        "|".join(re.escape(heading).replace(r"\ ", r"\s+") for heading in normalized_headings),
        re.IGNORECASE,
    )
    chunks: list[Chunk] = []
    chunk_number = 1
    current_section = "Introduction"

    for page_number, page_text in enumerate(pages, start=1):
        normalized_text = re.sub(r"\s+", " ", page_text).strip()
        matches = list(heading_pattern.finditer(normalized_text))
        boundaries = [0] + [match.start() for match in matches] + [len(normalized_text)]
        for index in range(len(boundaries) - 1):
            start, end = boundaries[index], boundaries[index + 1]
            section_text = normalized_text[start:end].strip()
            if not section_text:
                continue
            section = current_section
            if matches and index > 0:
                current_section = matches[index - 1].group(0)
                section = current_section
            for text in fixed_size_chunks(section_text, chunk_size=chunk_size, overlap=overlap):
                chunks.append(
                    {
                        "chunk_id": f"{document_id}_chunk_{chunk_number:03d}",
                        "document_id": document_id,
                        "document_name": document_name,
                        "page_number": page_number,
                        "section": section,
                        "text": text,
                    }
                )
                chunk_number += 1
    return chunks

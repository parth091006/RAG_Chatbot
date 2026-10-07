from __future__ import annotations

import hashlib
import re
from typing import TypedDict

from .strategies import fixed_size_chunks


class ChunkRequired(TypedDict):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str


class ChunkOptional(TypedDict, total=False):
    embedding: list[float]
    section: str
    chunk_text_hash: str
    secondary_char_ratio: float | None
    secondary_spans: list[tuple[int, int]] | None
    header_spans: list[tuple[int, int]] | None


class Chunk(ChunkRequired, ChunkOptional):
    pass


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
    pages: list[str | dict[str, object]],
    document_id: str,
    document_name: str,
    chunk_size: int = 250,
    overlap: int = 50,
) -> list[Chunk]:
    """Create stable, page-aware chunk records from extracted PDF pages."""
    chunks: list[Chunk] = []
    chunk_number = 1
    for page_number, page in enumerate(pages, start=1):
        if isinstance(page, str):
            page_text = page
            page_chunks = fixed_size_chunks(
                page_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            chunk_regions = [None] * len(page_chunks)
        else:
            page_text = str(page["text"])
            page_chunks, chunk_regions = _chunk_page_with_regions(
                page_text,
                page.get("region_spans", []),
                chunk_size=chunk_size,
                overlap=overlap,
            )

        for text, regions in zip(page_chunks, chunk_regions):
            chunk: Chunk = {
                "chunk_id": f"{document_id}_chunk_{chunk_number:03d}",
                "document_id": document_id,
                "document_name": document_name,
                "page_number": page_number,
                "text": text,
                "chunk_text_hash": hashlib.sha256(
                    " ".join(text.split()).encode("utf-8")
                ).hexdigest()[:12],
            }

            if regions is not None:
                secondary_spans, header_spans = regions
                secondary_length = sum(
                    end - start
                    for start, end in secondary_spans
                )
                chunk["secondary_char_ratio"] = (
                    secondary_length / len(text)
                    if text
                    else 0.0
                )
                chunk["secondary_spans"] = secondary_spans
                chunk["header_spans"] = header_spans

            chunks.append(chunk)
            chunk_number += 1
    return chunks


def _chunk_page_with_regions(
    page_text: str,
    region_spans: object,
    chunk_size: int,
    overlap: int,
) -> tuple[list[str], list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]]:
    words = [
        match.group(0)
        for match in re.finditer(r"\S+", page_text)
    ]
    word_spans = [
        (match.start(), match.end())
        for match in re.finditer(r"\S+", page_text)
    ]
    spans = region_spans if isinstance(region_spans, list) else []
    chunks: list[str] = []
    regions: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        text = " ".join(words[start:end])
        chunks.append(text)

        secondary_spans: list[tuple[int, int]] = []
        header_spans: list[tuple[int, int]] = []
        output_offset = 0

        for word_index in range(start, end):
            source_start, source_end = word_spans[word_index]
            output_start = output_offset
            output_end = output_start + len(words[word_index])
            output_offset = output_end + 1

            for span in spans:
                if not isinstance(span, dict):
                    continue
                if (
                    source_start < int(span["end"])
                    and source_end > int(span["start"])
                ):
                    if span.get("region") == "secondary":
                        secondary_spans.append(
                            (output_start, output_end)
                        )
                    if span.get("header_like"):
                        header_spans.append(
                            (output_start, output_end)
                        )

        regions.append((secondary_spans, header_spans))

        if end == len(words):
            break

    return chunks, regions


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

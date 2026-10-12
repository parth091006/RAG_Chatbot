from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence, TypedDict

from .strategies import (
    ChunkQualityMetrics,
    deduplicate_chunks,
    fixed_size_chunks,
    merge_cross_page_text,
    recursive_chunks,
    semantic_chunks,
)


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


# ---------------------------------------------------------------------------
# Dynamic heading detection
# ---------------------------------------------------------------------------

# Pattern matches the start of a numbered academic heading:
# a number prefix like "2.1" followed by a capitalized word.
# The full heading text is extracted by _extract_heading_text().
_HEADING_START = re.compile(
    r"\b(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z]*)"
)


def _extract_heading_text(text: str, match_start: int, number_prefix: str) -> str:
    """Extract the full heading text starting from a numbered heading match.

    Reads forward from the match position until it encounters clear body
    text signals: a sentence with a verb phrase, a period followed by
    a space and lowercase letter, or exceeds 80 characters.
    """
    # Start from the match position
    remaining = text[match_start:]
    # Find the end of the heading: stop at the first sentence-ending
    # punctuation followed by a space, or after ~80 chars
    heading_end = len(remaining)
    for i, ch in enumerate(remaining):
        if i > 80:
            # Too long for a heading — truncate at last word boundary
            heading_end = remaining.rfind(" ", 0, i)
            if heading_end <= len(number_prefix) + 1:
                heading_end = i
            break
        if ch in ".!?" and i + 1 < len(remaining) and remaining[i + 1] == " ":
            # Check if what follows looks like body text
            after = remaining[i + 2: i + 12].strip()
            if after and after[0].islower():
                heading_end = i
                break
            # Uppercase after period could be next sentence in heading
            # (unlikely but stop anyway for safety)
            heading_end = i
            break
    return remaining[:heading_end].strip()


_COMMON_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "methodology",
    "methods",
    "method",
    "approach",
    "architecture",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "appendix",
    "acknowledgments",
    "acknowledgements",
}


def detect_headings(text: str) -> list[str]:
    """Dynamically detect section headings from text.

    Uses both numbered academic heading patterns and a set of common
    section names.  Returns headings in the order they appear.
    """
    found: list[tuple[int, str]] = []

    # Numbered headings
    for match in _NUMBERED_HEADING.finditer(text):
        heading = match.group(0).strip()
        # Reject if the text after the number is very short (likely a
        # figure/table number) or all digits
        label = match.group(2).strip()
        if len(label) < 3 or label.replace(" ", "").isdigit():
            continue
        found.append((match.start(), heading))

    # Common un-numbered section headings (standalone lines)
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower() in _COMMON_SECTIONS:
            pos = text.find(line)
            if pos >= 0:
                found.append((pos, stripped))

    # Sort by position and deduplicate
    found.sort(key=lambda x: x[0])
    seen: set[str] = set()
    result: list[str] = []
    for _, heading in found:
        normalized = " ".join(heading.split())
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


# ---------------------------------------------------------------------------
# Core chunking functions
# ---------------------------------------------------------------------------

def chunk_document(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    strategy: str = "fixed",
) -> list[str]:
    """Split a document text into chunks using the specified chunking strategy."""
    if not text:
        return []
    if strategy == "recursive":
        return recursive_chunks(text, chunk_size=chunk_size, overlap=overlap)
    if strategy == "semantic":
        return semantic_chunks(text, chunk_size=chunk_size, overlap=overlap)
    return fixed_size_chunks(text, chunk_size=chunk_size, overlap=overlap)


def chunk_pages(
    pages: list[str | dict[str, object]],
    document_id: str,
    document_name: str,
    chunk_size: int = 250,
    overlap: int = 50,
    cross_page_merge: bool = False,
) -> list[Chunk]:
    """Create stable, page-aware chunk records from extracted PDF pages."""
    # Optionally merge text across page boundaries
    page_texts = [_page_text(p) for p in pages]
    if cross_page_merge:
        page_texts = merge_cross_page_text(page_texts)

    chunks: list[Chunk] = []
    chunk_number = 1
    for page_number, (page, merged_text) in enumerate(
        zip(pages, page_texts), start=1
    ):
        if isinstance(page, str):
            page_chunks = fixed_size_chunks(
                merged_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            chunk_regions = [None] * len(page_chunks)
        else:
            page_chunks, chunk_regions = _chunk_page_with_regions(
                merged_text,
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


def recursive_page_chunks(
    pages: list[str | dict[str, object]],
    document_id: str,
    document_name: str,
    chunk_size: int = 250,
    overlap: int = 50,
    cross_page_merge: bool = False,
) -> list[Chunk]:
    """Create page-aware chunks using hierarchical recursive splitting."""
    page_texts = [_page_text(p) for p in pages]
    if cross_page_merge:
        page_texts = merge_cross_page_text(page_texts)

    chunks: list[Chunk] = []
    chunk_number = 1
    for page_number, merged_text in enumerate(page_texts, start=1):
        page_chunks = recursive_chunks(
            merged_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        for text in page_chunks:
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
            chunks.append(chunk)
            chunk_number += 1
    return chunks


def semantic_page_chunks(
    pages: list[str | dict[str, object]],
    document_id: str,
    document_name: str,
    chunk_size: int = 250,
    overlap: int = 50,
    similarity_threshold: float = 0.15,
    cross_page_merge: bool = True,
) -> list[Chunk]:
    """Create page-aware chunks using semantic (topic-shift) splitting.

    By default, cross-page merging is enabled since semantic chunking
    benefits most from unbroken text flow.
    """
    page_texts = [_page_text(p) for p in pages]
    if cross_page_merge:
        page_texts = merge_cross_page_text(page_texts)

    chunks: list[Chunk] = []
    chunk_number = 1
    for page_number, merged_text in enumerate(page_texts, start=1):
        if not merged_text.strip():
            continue
        page_chunks = semantic_chunks(
            merged_text,
            chunk_size=chunk_size,
            overlap=overlap,
            similarity_threshold=similarity_threshold,
        )
        for text in page_chunks:
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
            chunks.append(chunk)
            chunk_number += 1
    return chunks


def section_aware_chunks(
    pages: list[str | dict[str, object]],
    document_id: str,
    document_name: str,
    chunk_size: int = 250,
    overlap: int = 50,
    headings: tuple[str, ...] | list[str] | None = None,
    cross_page_merge: bool = False,
) -> list[Chunk]:
    """Split pages at section headings, then chunk within each section.

    When *headings* is ``None`` the function dynamically detects headings
    using both numbered academic patterns and common section names.
    """
    page_texts = [_page_text(p) for p in pages]
    if cross_page_merge:
        page_texts = merge_cross_page_text(page_texts)

    # Build heading detection pattern
    if headings is not None:
        target_headings = list(headings)
    else:
        # Dynamically detect from combined text
        full_text = "\n\n".join(page_texts)
        target_headings = detect_headings(full_text)

    normalized_headings = {
        re.sub(r"\s+", " ", heading).strip(): heading for heading in target_headings
    }
    exact_patterns = [
        re.escape(heading).replace(r"\ ", r"\s+") for heading in normalized_headings
    ]
    # Dynamic academic numbered heading pattern (e.g., "1. Introduction", "2.1 Background")
    # Matches number prefix + sequence of capitalized words only.
    dynamic_pattern = r"\b\d+(?:\.\d+)*\s+(?:[A-Z][A-Za-z0-9—\-:,]*(?:\s+|$))+"
    all_patterns = exact_patterns + [dynamic_pattern]
    heading_pattern = re.compile("|".join(all_patterns), re.IGNORECASE)

    chunks: list[Chunk] = []
    chunk_number = 1
    current_section = "Introduction"

    for page_number, merged_text in enumerate(page_texts, start=1):
        normalized_text = re.sub(r"\s+", " ", merged_text).strip()
        if not normalized_text:
            continue
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


# ---------------------------------------------------------------------------
# Chunk deduplication for Chunk dicts
# ---------------------------------------------------------------------------

def deduplicate_chunk_records(
    chunks: list[Chunk],
    similarity_threshold: float = 0.85,
) -> list[Chunk]:
    """Remove near-duplicate Chunk records.

    Works like ``deduplicate_chunks`` from strategies but operates on
    Chunk dicts and re-assigns sequential chunk IDs after deduplication.
    """
    if not chunks:
        return []

    texts = [c["text"] for c in chunks]
    deduped_texts = deduplicate_chunks(texts, similarity_threshold=similarity_threshold)
    deduped_set = set(deduped_texts)

    result: list[Chunk] = []
    seen_texts: set[str] = set()
    for chunk in chunks:
        if chunk["text"] in deduped_set and chunk["text"] not in seen_texts:
            result.append(chunk)
            seen_texts.add(chunk["text"])

    # Re-assign sequential chunk IDs
    if result:
        doc_id = result[0]["document_id"]
        for i, chunk in enumerate(result, start=1):
            chunk["chunk_id"] = f"{doc_id}_chunk_{i:03d}"

    return result


# ---------------------------------------------------------------------------
# Chunk quality report
# ---------------------------------------------------------------------------

def compute_chunk_quality(chunks: list[Chunk]) -> dict[str, float | int]:
    """Compute quality metrics for a list of Chunk records."""
    texts = [c["text"] for c in chunks]
    return ChunkQualityMetrics(texts).to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page_text(page: str | dict[str, object]) -> str:
    """Extract plain text from a page (string or region dict)."""
    if isinstance(page, str):
        return page
    return str(page.get("text", ""))

from __future__ import annotations

import hashlib
import re
from typing import Sequence


def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks: list[str] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break

    return chunks


def recursive_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    """Hierarchically split text using a prioritized list of separators.

    Separators are tried in priority order (default: paragraphs, newlines,
    sentence boundaries, words). Small segments are recombined up to
    chunk_size (in words) while preserving overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    words = cleaned.split()
    if len(words) <= chunk_size:
        return [cleaned]

    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _split_into_atomic_pieces(content: str, sep_index: int) -> list[str]:
        content = content.strip()
        if not content:
            return []
        c_words = content.split()
        if len(c_words) <= chunk_size:
            return [content]
        if sep_index >= len(separators):
            return fixed_size_chunks(content, chunk_size=chunk_size, overlap=overlap)

        sep = separators[sep_index]
        if sep in content:
            raw_parts = content.split(sep)
            result = []
            for part in raw_parts:
                part = part.strip()
                if not part:
                    continue
                if sep == ". ":
                    part = part if part.endswith(".") else part + "."
                if len(part.split()) > chunk_size:
                    result.extend(_split_into_atomic_pieces(part, sep_index + 1))
                else:
                    result.append(part)
            return result
        return _split_into_atomic_pieces(content, sep_index + 1)

    pieces = _split_into_atomic_pieces(cleaned, 0)
    if not pieces:
        return []

    chunks: list[str] = []
    current_pieces: list[str] = []
    current_words = 0

    for piece in pieces:
        piece_words = len(piece.split())
        if current_pieces and (current_words + piece_words > chunk_size):
            chunk_text = " ".join(current_pieces).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Build overlap pieces from tail of current_pieces
            overlap_pieces: list[str] = []
            overlap_words = 0
            for prev in reversed(current_pieces):
                w_count = len(prev.split())
                if overlap_words + w_count <= overlap:
                    overlap_pieces.insert(0, prev)
                    overlap_words += w_count
                else:
                    break
            current_pieces = overlap_pieces
            current_words = overlap_words

        current_pieces.append(piece)
        current_words += piece_words

    if current_pieces:
        chunk_text = " ".join(current_pieces).strip()
        if chunk_text and (not chunks or chunk_text != chunks[-1]):
            chunks.append(chunk_text)

    return chunks


# ---------------------------------------------------------------------------
# Sentence splitting utility
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(
    r"""
    (?<=[.!?])          # preceded by sentence-ending punctuation
    (?:\s+|$)           # followed by whitespace or end of string
    """,
    re.VERBOSE,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based heuristics.

    Handles abbreviations and decimal numbers reasonably well by requiring
    whitespace after the sentence-end punctuation.
    """
    parts = _SENTENCE_END.split(text.strip())
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


# ---------------------------------------------------------------------------
# Semantic Chunking
# ---------------------------------------------------------------------------

def _word_set_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between word sets of two strings.

    This is a lightweight, embedding-free proxy for semantic similarity.
    It works well enough for detecting topic shifts within a single document
    without requiring any model downloads.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def semantic_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    similarity_threshold: float = 0.15,
) -> list[str]:
    """Split text into chunks at points where topical similarity drops.

    Uses sentence-level Jaccard similarity as a lightweight semantic
    signal.  When the similarity between consecutive sentence windows
    drops below *similarity_threshold*, a chunk boundary is inserted.

    Sentences are then recombined into chunks that respect *chunk_size*
    (in words) and *overlap*.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    sentences = _split_sentences(cleaned)
    if not sentences:
        return [cleaned]

    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= chunk_size:
        return [cleaned]

    # Identify break points using similarity between sliding windows of
    # sentences.  A window of 3 sentences captures enough local context.
    window = 3
    break_indices: list[int] = []

    for i in range(window, len(sentences)):
        left = " ".join(sentences[max(0, i - window): i])
        right = " ".join(sentences[i: min(len(sentences), i + window)])
        sim = _word_set_similarity(left, right)
        if sim < similarity_threshold:
            break_indices.append(i)

    # Build raw semantic segments from break indices
    segments: list[str] = []
    prev = 0
    for idx in break_indices:
        segment = " ".join(sentences[prev:idx]).strip()
        if segment:
            segments.append(segment)
        prev = idx
    tail = " ".join(sentences[prev:]).strip()
    if tail:
        segments.append(tail)

    if not segments:
        return [cleaned]

    # Recombine segments into chunks respecting chunk_size with overlap
    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for segment in segments:
        seg_words = len(segment.split())
        if current_parts and current_words + seg_words > chunk_size:
            chunk_text = " ".join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Build overlap from trailing current_parts
            overlap_parts: list[str] = []
            overlap_word_count = 0
            for prev_part in reversed(current_parts):
                w = len(prev_part.split())
                if overlap_word_count + w <= overlap:
                    overlap_parts.insert(0, prev_part)
                    overlap_word_count += w
                else:
                    break
            current_parts = overlap_parts
            current_words = overlap_word_count

        # If a single segment exceeds chunk_size, split it further
        if seg_words > chunk_size:
            sub_chunks = fixed_size_chunks(segment, chunk_size=chunk_size, overlap=overlap)
            for sc in sub_chunks:
                chunks.append(sc)
            current_parts = []
            current_words = 0
        else:
            current_parts.append(segment)
            current_words += seg_words

    if current_parts:
        chunk_text = " ".join(current_parts).strip()
        if chunk_text and (not chunks or chunk_text != chunks[-1]):
            chunks.append(chunk_text)

    return chunks


# ---------------------------------------------------------------------------
# Cross-page boundary merging
# ---------------------------------------------------------------------------

def merge_cross_page_text(
    pages: Sequence[str],
    min_continuation_words: int = 3,
) -> list[str]:
    """Merge text across page boundaries when a sentence spans pages.

    If a page ends without sentence-ending punctuation and the next page
    starts with a lowercase letter or a short continuation, the boundary
    text is merged.  This prevents mid-sentence chunk breaks at page
    boundaries.

    Returns a list with the same length as *pages* but with merged text
    moved to the earlier page and removed from the later page.
    """
    if not pages:
        return []
    if len(pages) == 1:
        return list(pages)

    merged: list[str] = [pages[0]]

    for i in range(1, len(pages)):
        prev_text = merged[-1].rstrip()
        curr_text = pages[i].lstrip()

        if not prev_text or not curr_text:
            merged.append(pages[i])
            continue

        # Check if the previous page likely ends mid-sentence
        ends_mid_sentence = not prev_text[-1] in ".!?:;\""
        # Check if the current page starts with a continuation pattern
        starts_continuation = bool(
            re.match(r"^[a-z]", curr_text)
        ) or (
            len(curr_text.split()) >= min_continuation_words
            and not re.match(r"^\d+(\.\d+)*\s+[A-Z]", curr_text)
            and not re.match(r"^(Abstract|Introduction|Conclusion|References|Appendix|Table|Figure)\b", curr_text, re.IGNORECASE)
        )

        if ends_mid_sentence and starts_continuation:
            # Find the end of the first sentence on the current page
            first_sentence_end = None
            for m in _SENTENCE_END.finditer(curr_text):
                first_sentence_end = m.end()
                break

            if first_sentence_end is not None:
                # Move the first sentence fragment to the previous page
                carry_over = curr_text[:first_sentence_end].strip()
                remainder = curr_text[first_sentence_end:].strip()
                merged[-1] = prev_text + " " + carry_over
                merged.append(remainder if remainder else "")
            else:
                # Entire current page is a continuation — merge fully
                merged[-1] = prev_text + " " + curr_text
                merged.append("")
        else:
            merged.append(pages[i])

    return merged


# ---------------------------------------------------------------------------
# Chunk deduplication
# ---------------------------------------------------------------------------

def _chunk_text_hash(text: str) -> str:
    """Hash normalized chunk text for deduplication."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def deduplicate_chunks(
    chunks: list[str],
    similarity_threshold: float = 0.85,
) -> list[str]:
    """Remove near-duplicate chunks that arise from high overlap.

    Exact duplicates (by normalized hash) are always removed.
    Chunks with Jaccard similarity ≥ *similarity_threshold* to an
    already-accepted chunk are also dropped.
    """
    if not chunks:
        return []

    seen_hashes: set[str] = set()
    accepted: list[str] = []

    for chunk in chunks:
        h = _chunk_text_hash(chunk)
        if h in seen_hashes:
            continue

        # Check near-duplicate against recent accepted chunks (window of 5)
        is_near_dup = False
        for recent in accepted[-5:]:
            if _word_set_similarity(chunk, recent) >= similarity_threshold:
                is_near_dup = True
                break

        if not is_near_dup:
            seen_hashes.add(h)
            accepted.append(chunk)

    return accepted


# ---------------------------------------------------------------------------
# Chunk quality metrics
# ---------------------------------------------------------------------------

class ChunkQualityMetrics:
    """Statistics about a set of chunks useful for tuning parameters."""

    def __init__(self, chunks: Sequence[str]) -> None:
        self.count = len(chunks)
        word_counts = [len(c.split()) for c in chunks] if chunks else [0]
        self.total_words = sum(word_counts)
        self.min_words = min(word_counts) if word_counts else 0
        self.max_words = max(word_counts) if word_counts else 0
        self.avg_words = self.total_words / max(self.count, 1)
        # Fragmentation ratio: proportion of very short chunks (< 30 words)
        short = sum(1 for wc in word_counts if wc < 30)
        self.fragmentation_ratio = short / max(self.count, 1)
        # Overlap ratio: approximate by looking at consecutive pairs
        overlap_scores: list[float] = []
        for i in range(len(chunks) - 1):
            overlap_scores.append(_word_set_similarity(chunks[i], chunks[i + 1]))
        self.avg_overlap = (
            sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_words": self.total_words,
            "min_words": self.min_words,
            "max_words": self.max_words,
            "avg_words": round(self.avg_words, 1),
            "fragmentation_ratio": round(self.fragmentation_ratio, 3),
            "avg_overlap": round(self.avg_overlap, 3),
        }

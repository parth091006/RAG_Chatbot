"""Tests for Step 2: Advanced Chunking.

Covers:
- Semantic chunking (topic-shift detection)
- Cross-page boundary merging
- Chunk deduplication
- Dynamic heading detection
- Chunk quality metrics
- Pipeline integration with new strategies
"""
from __future__ import annotations

import pytest

from src.rag.chunking.chunker import (
    Chunk,
    chunk_document,
    chunk_pages,
    compute_chunk_quality,
    deduplicate_chunk_records,
    detect_headings,
    recursive_page_chunks,
    section_aware_chunks,
    semantic_page_chunks,
)
from src.rag.chunking.strategies import (
    ChunkQualityMetrics,
    _split_sentences,
    _word_set_similarity,
    deduplicate_chunks,
    merge_cross_page_text,
    semantic_chunks,
)


# -----------------------------------------------------------------------
# Semantic chunking tests
# -----------------------------------------------------------------------

class TestSemanticChunks:
    def test_returns_list(self) -> None:
        text = "Machine learning is a branch of AI. " * 100
        result = semantic_chunks(text, chunk_size=30, overlap=5)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_text(self) -> None:
        assert semantic_chunks("", chunk_size=50, overlap=10) == []

    def test_short_text_returns_single_chunk(self) -> None:
        text = "Short sentence about transformers."
        result = semantic_chunks(text, chunk_size=100, overlap=10)
        assert len(result) == 1

    def test_topic_shift_creates_boundary(self) -> None:
        """Text with two clearly different topics should produce chunks
        that respect the topic boundary."""
        topic_a = "Neural networks use backpropagation for gradient descent training. " * 20
        topic_b = "The economy grew by five percent this quarter in banking. " * 20
        text = topic_a + topic_b
        result = semantic_chunks(text, chunk_size=50, overlap=5, similarity_threshold=0.10)
        assert len(result) >= 2

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError):
            semantic_chunks("text", chunk_size=0, overlap=0)
        with pytest.raises(ValueError):
            semantic_chunks("text", chunk_size=10, overlap=-1)
        with pytest.raises(ValueError):
            semantic_chunks("text", chunk_size=10, overlap=10)


class TestSentenceSplitting:
    def test_basic_split(self) -> None:
        text = "First sentence. Second sentence. Third one."
        result = _split_sentences(text)
        assert len(result) >= 2

    def test_empty_text(self) -> None:
        assert _split_sentences("") == []


class TestWordSetSimilarity:
    def test_identical_strings(self) -> None:
        assert _word_set_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert _word_set_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _word_set_similarity("hello world foo", "hello world bar")
        assert 0.0 < sim < 1.0

    def test_empty_string(self) -> None:
        assert _word_set_similarity("", "hello") == 0.0


# -----------------------------------------------------------------------
# Cross-page boundary merging tests
# -----------------------------------------------------------------------

class TestCrossPageMerge:
    def test_no_merge_on_complete_sentences(self) -> None:
        pages = [
            "This is page one. It ends properly.",
            "This is page two. It also ends properly.",
        ]
        result = merge_cross_page_text(pages)
        assert result[0] == pages[0]
        assert result[1] == pages[1]

    def test_merge_mid_sentence(self) -> None:
        pages = [
            "The transformer architecture uses",
            "multi-head attention for parallel processing.",
        ]
        result = merge_cross_page_text(pages)
        # First page should absorb the continuation
        assert "multi-head attention" in result[0]

    def test_no_merge_on_new_section(self) -> None:
        pages = [
            "End of previous section content",
            "3.1 Multi-head Self-Attention This section describes...",
        ]
        result = merge_cross_page_text(pages)
        # Should NOT merge because page 2 starts with a section heading
        assert result[1].startswith("3.1")

    def test_empty_pages(self) -> None:
        assert merge_cross_page_text([]) == []
        assert merge_cross_page_text(["hello"]) == ["hello"]

    def test_preserves_page_count(self) -> None:
        pages = ["Page one text", "page two continues", "Page three starts fresh."]
        result = merge_cross_page_text(pages)
        assert len(result) == len(pages)


# -----------------------------------------------------------------------
# Deduplication tests
# -----------------------------------------------------------------------

class TestDeduplication:
    def test_exact_duplicate_removed(self) -> None:
        chunks = ["Hello world", "Hello world", "Different chunk"]
        result = deduplicate_chunks(chunks)
        assert len(result) == 2

    def test_near_duplicate_removed(self) -> None:
        chunks = [
            "The transformer model uses attention mechanisms for processing",
            "The transformer model uses attention mechanisms for processing input",
            "A completely different topic about economics and finance",
        ]
        result = deduplicate_chunks(chunks, similarity_threshold=0.80)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        assert deduplicate_chunks([]) == []

    def test_all_unique(self) -> None:
        chunks = ["Alpha beta gamma", "Delta epsilon zeta", "Eta theta iota"]
        result = deduplicate_chunks(chunks)
        assert len(result) == 3


class TestDeduplicateChunkRecords:
    def _make_chunk(self, text: str, idx: int) -> Chunk:
        return {
            "chunk_id": f"doc_test_chunk_{idx:03d}",
            "document_id": "doc_test",
            "document_name": "test.pdf",
            "page_number": 1,
            "text": text,
        }

    def test_removes_duplicate_records(self) -> None:
        chunks = [
            self._make_chunk("Hello world foo bar", 1),
            self._make_chunk("Hello world foo bar", 2),
            self._make_chunk("Different content entirely", 3),
        ]
        result = deduplicate_chunk_records(chunks)
        assert len(result) == 2
        # IDs should be re-assigned sequentially
        assert result[0]["chunk_id"] == "doc_test_chunk_001"
        assert result[1]["chunk_id"] == "doc_test_chunk_002"

    def test_empty_input(self) -> None:
        assert deduplicate_chunk_records([]) == []


# -----------------------------------------------------------------------
# Dynamic heading detection tests
# -----------------------------------------------------------------------

class TestDetectHeadings:
    def test_numbered_headings(self) -> None:
        text = """
        1 Introduction
        This paper presents a new approach.
        2.1 Background and Related Work
        Prior work has shown significant progress.
        3 Methodology
        We propose the following method.
        """
        headings = detect_headings(text)
        assert any("Introduction" in h for h in headings)
        assert any("Background" in h for h in headings)
        assert any("Methodology" in h for h in headings)

    def test_common_section_names(self) -> None:
        text = "Abstract\nThis paper discusses...\nConclusion\nWe have shown..."
        headings = detect_headings(text)
        lower_headings = [h.lower() for h in headings]
        assert "abstract" in lower_headings
        assert "conclusion" in lower_headings

    def test_empty_text(self) -> None:
        assert detect_headings("") == []

    def test_no_headings(self) -> None:
        text = "just some regular text without any headings at all"
        headings = detect_headings(text)
        # May find some false positives but should be minimal
        assert isinstance(headings, list)


# -----------------------------------------------------------------------
# Chunk quality metrics tests
# -----------------------------------------------------------------------

class TestChunkQualityMetrics:
    def test_basic_metrics(self) -> None:
        chunks = [
            "word " * 50,
            "word " * 100,
            "word " * 20,
        ]
        metrics = ChunkQualityMetrics(chunks)
        assert metrics.count == 3
        assert metrics.min_words == 20
        assert metrics.max_words == 100
        assert metrics.avg_words == pytest.approx(170 / 3, abs=0.5)
        assert 0.0 <= metrics.fragmentation_ratio <= 1.0
        assert 0.0 <= metrics.avg_overlap <= 1.0

    def test_empty_chunks(self) -> None:
        metrics = ChunkQualityMetrics([])
        assert metrics.count == 0

    def test_to_dict(self) -> None:
        chunks = ["hello world", "foo bar baz"]
        d = ChunkQualityMetrics(chunks).to_dict()
        assert "count" in d
        assert "total_words" in d
        assert "fragmentation_ratio" in d
        assert "avg_overlap" in d


class TestComputeChunkQuality:
    def test_returns_dict(self) -> None:
        chunks: list[Chunk] = [
            {
                "chunk_id": "c_001",
                "document_id": "doc",
                "document_name": "test.pdf",
                "page_number": 1,
                "text": "word " * 50,
            }
        ]
        result = compute_chunk_quality(chunks)
        assert isinstance(result, dict)
        assert result["count"] == 1


# -----------------------------------------------------------------------
# chunk_document with semantic strategy
# -----------------------------------------------------------------------

class TestChunkDocumentSemantic:
    def test_semantic_strategy(self) -> None:
        text = " ".join(f"word_{i}" for i in range(500))
        result = chunk_document(text, chunk_size=50, overlap=10, strategy="semantic")
        assert isinstance(result, list)
        assert len(result) > 0


# -----------------------------------------------------------------------
# Page-level chunking with cross-page merge
# -----------------------------------------------------------------------

class TestPageChunksWithCrossPageMerge:
    def test_cross_page_merge_flag(self) -> None:
        pages = [
            "The transformer architecture uses",
            "multi-head attention for parallel processing. This is important.",
        ]
        chunks_no_merge = chunk_pages(
            pages,
            document_id="doc_test",
            document_name="test.pdf",
            chunk_size=50,
            overlap=5,
            cross_page_merge=False,
        )
        chunks_with_merge = chunk_pages(
            pages,
            document_id="doc_test",
            document_name="test.pdf",
            chunk_size=50,
            overlap=5,
            cross_page_merge=True,
        )
        # Both should produce chunks but content may differ
        assert len(chunks_no_merge) > 0
        assert len(chunks_with_merge) > 0


# -----------------------------------------------------------------------
# Semantic page chunks
# -----------------------------------------------------------------------

class TestSemanticPageChunks:
    def test_produces_chunks(self) -> None:
        pages = [
            "Neural networks learn representations through backpropagation. " * 10,
            "Economic growth in developing nations has accelerated. " * 10,
        ]
        chunks = semantic_page_chunks(
            pages,
            document_id="doc_test",
            document_name="test.pdf",
            chunk_size=30,
            overlap=5,
        )
        assert len(chunks) > 0
        assert all(c["document_id"] == "doc_test" for c in chunks)


# -----------------------------------------------------------------------
# Section-aware chunks with dynamic detection
# -----------------------------------------------------------------------

class TestSectionAwareDynamic:
    def test_dynamic_heading_detection(self) -> None:
        """When headings=None, section_aware_chunks should detect headings
        dynamically instead of using hardcoded section names."""
        pages = [
            "1 Introduction This paper presents a novel approach to chunking.",
            "2 Methodology We use the following steps to process documents.",
            "3 Results Our experiments show significant improvement.",
        ]
        chunks = section_aware_chunks(
            pages,
            document_id="doc_test",
            document_name="test.pdf",
            chunk_size=50,
            overlap=5,
            headings=None,
        )
        assert len(chunks) >= 3
        sections = [c.get("section", "") for c in chunks]
        assert any("Introduction" in s for s in sections)

    def test_explicit_headings_still_work(self) -> None:
        """Passing explicit headings should still function correctly."""
        pages = [
            "Custom Header One Some content here about the topic.",
            "Custom Header Two More content about another topic.",
        ]
        chunks = section_aware_chunks(
            pages,
            document_id="doc_test",
            document_name="test.pdf",
            chunk_size=50,
            overlap=5,
            headings=["Custom Header One", "Custom Header Two"],
        )
        assert len(chunks) >= 2

from src.rag.generation.generator import Generator
from scripts.check_ground_truth import (
    header_chunk_counts,
    overlap,
    remap_chunks,
    word_difference,
)


def test_word_difference_and_overlap_are_deterministic() -> None:
    dropped = word_difference(
        "attention improves parallel computation",
        "attention improves computation",
    )

    assert dropped == "parallel"
    assert overlap(
        "parallel computation",
        dropped,
    ) == 0.5


def test_remap_reports_best_and_second_best_matches() -> None:
    source = {
        "source_001": {"text": "attention uses query key value"},
    }
    target = {
        "target_001": {"text": "attention uses query key value"},
        "target_002": {"text": "attention uses query and key"},
    }

    result = remap_chunks(source, target, "source", "target")

    assert result["mappings"][0]["target_chunk_id"] == "target_001"
    assert result["mappings"][0]["jaccard"] == 1.0
    assert result["mappings"][0]["second_best_target_chunk_id"] == "target_002"


def test_header_chunk_counts_and_generator_tokenization() -> None:
    chunks = [
        {"text": "with deep learning 2 context"},
        {"text": "context with deep learning 3"},
        {"text": "clean context"},
    ]

    counts = header_chunk_counts(chunks)

    assert counts["chunks_containing_leaked_header"] == 2
    assert counts["chunks_starting_with_leaked_header"] == 1
    assert counts["chunks_ending_with_leaked_header"] == 1
    assert Generator._extract_terms("Self-attention uses keys") == {
        "self-attention",
        "uses",
        "keys",
    }

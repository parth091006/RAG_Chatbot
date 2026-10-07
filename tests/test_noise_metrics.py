from pathlib import Path
import re

from src.rag.evaluation.noise_metrics import (
    answer_evidence_support,
    answer_evidence_support_variants,
    answer_fragment_diagnostics,
    answer_header_leak,
    answer_secondary_ratio,
    answer_unlocatable_sentence_count,
    context_noise_per_chunk,
    document_label_like_stats,
    diagram_noise_ratio,
    header_leak,
    header_leak_counts,
    label_like_char_share,
    label_like_fragment,
)
from src.rag.ingestion.parser import (
    extract_pages_from_pdf,
    extract_pages_with_regions,
)
from src.rag.chunking.chunker import chunk_pages
from src.rag.pipeline import PhaseOneRAG


PDF_PATH = Path("data/raw/self-attention-transformers-2023.pdf")


def test_region_provenance_and_chunk_text_are_stable() -> None:
    pages = extract_pages_with_regions(
        PDF_PATH,
        include_secondary=True,
        strip_page_number_headers=False,
    )
    plain_pages = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=True,
        strip_page_number_headers=False,
    )

    assert [page["text"] for page in pages] == plain_pages
    assert len(pages) == 18
    assert sum(
        span["header_like"]
        for page in pages
        for span in page["region_spans"]
    ) == 17
    assert sum(
        len(page["removed_blocks"])
        for page in pages
    ) == 18

    plain_chunks = chunk_pages(plain_pages, "doc_test", "paper.pdf")
    region_chunks = chunk_pages(pages, "doc_test", "paper.pdf")
    assert [chunk["text"] for chunk in region_chunks] == [
        chunk["text"] for chunk in plain_chunks
    ]
    assert any(
        chunk.get("secondary_char_ratio") is not None
        for chunk in region_chunks
    )


def test_header_fix_removes_header_like_output_spans() -> None:
    pages = extract_pages_with_regions(
        PDF_PATH,
        include_secondary=True,
        strip_page_number_headers=True,
    )

    assert sum(
        span["header_like"]
        for page in pages
        for span in page["region_spans"]
    ) == 0
    assert sum(
        bool(
            re.fullmatch(
                r"with deep learning \d{1,3}",
                " ".join(block["text"].casefold().split()),
            )
        )
        for page in pages
        for block in page["removed_blocks"]
    ) == 17


def test_legacy_noise_metrics_are_null() -> None:
    chunk = {"text": "legacy text"}

    assert header_leak(chunk) is None
    assert diagram_noise_ratio([chunk]) is None
    assert context_noise_per_chunk([chunk]) == {
        "values": [None],
        "mean": None,
        "null_excluded": 1,
    }
    assert header_leak_counts([chunk])["count"] is None
    assert answer_secondary_ratio("legacy text", [chunk]) is None
    assert answer_header_leak("legacy text", [chunk]) is None


def test_noise_metrics_use_spans_and_real_answers() -> None:
    contexts = [
        {
            "text": "clean evidence sentence.",
            "secondary_char_ratio": 0.25,
            "secondary_spans": [(0, 5)],
            "header_spans": [],
        },
        {
            "text": "with deep learning 2.",
            "secondary_char_ratio": 0.75,
            "secondary_spans": [(0, 21)],
            "header_spans": [(0, 21)],
        },
    ]

    assert header_leak(contexts[0]) is False
    assert header_leak(contexts[1]) is True
    assert header_leak_counts(contexts) == {
        "count": 1,
        "total": 2,
        "null": 0,
    }
    assert diagram_noise_ratio(contexts) == 0.5
    assert context_noise_per_chunk(contexts) == {
        "values": [0.25, 0.75],
        "mean": 0.5,
        "null_excluded": 0,
    }
    assert answer_secondary_ratio(
        "with deep learning 2.",
        contexts,
    ) == 1.0
    assert answer_header_leak(
        "with deep learning 2.",
        contexts,
    ) is True
    assert answer_unlocatable_sentence_count(
        "missing sentence.",
        contexts,
    ) == 1
    assert answer_evidence_support(
        "clean evidence sentence.",
        contexts,
    ) == 1.0


def test_pipeline_legacy_has_no_region_metrics() -> None:
    pipeline = PhaseOneRAG(PDF_PATH)
    chunks = pipeline.pdf_to_chunks()

    assert chunks
    assert all(
        "secondary_char_ratio" not in chunk
        for chunk in chunks
    )


def test_spliced_sentence_is_reported_and_supported_by_fragments() -> None:
    contexts = [
        {
            "chunk_id": "chunk_a",
            "text": "first fragment without punctuation",
            "secondary_char_ratio": 0.0,
            "secondary_spans": [],
            "header_spans": [],
        },
        {
            "chunk_id": "chunk_b",
            "text": "second fragment.",
            "secondary_char_ratio": 1.0,
            "secondary_spans": [(0, 15)],
            "header_spans": [],
        },
    ]
    answer = "first fragment without punctuation second fragment."

    diagnostics = answer_fragment_diagnostics(
        answer,
        contexts,
    )

    assert diagnostics["spliced_sentence_count"] == 1
    assert diagnostics["sentences"][0]["spliced"] is True
    assert diagnostics["unlocated_character_share"] == 0.0
    assert answer_evidence_support_variants(
        answer,
        contexts,
    ) == {
        "whole_sentence": 0.0,
        "fragment_level": 1.0,
    }


def test_label_like_diagnostic_flags_label_runs_not_prose() -> None:
    label_run = "Add & Norm Multi-Head Attention Repeat for number of encoder blocks"
    prose = "This ordinary sentence explains how the encoder processes the input sequence."

    assert label_like_fragment(label_run, threshold=0.25) is True
    assert label_like_fragment(prose, threshold=0.25) is False
    assert document_label_like_stats(
        [{"text": label_run}, {"text": prose}]
    )["flagged_fragment_count"] == 1
    assert label_like_char_share(
        label_run,
        [{"text": label_run}],
    ) == 1.0

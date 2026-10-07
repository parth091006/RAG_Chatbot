import json
import re
from pathlib import Path

import pytest

from src.rag.ingestion.parser import extract_pages_from_pdf
from src.rag.pipeline import PhaseOneRAG


PDF_PATH = Path("data/raw/self-attention-transformers-2023.pdf")
CURRENT_SNAPSHOT = Path(
    "tests/fixtures/parser_current_snapshot.json"
)
HEADER_FIX_SNAPSHOT = Path(
    "tests/fixtures/parser_current_headerfix_snapshot.json"
)
HEADER_PATTERN = re.compile(
    r"^with deep learning(?: \d{1,3})?$"
)


def _normalized_lines(pages: list[str]) -> list[str]:
    return [
        " ".join(line.casefold().split())
        for page in pages
        for line in page.splitlines()
    ]


def _body_phrase_count(pages: list[str]) -> int:
    return sum(
        1
        for line in _normalized_lines(pages)
        if "with deep learning" in line
        and not HEADER_PATTERN.fullmatch(line)
    )


def test_header_fix_off_remains_equal_to_existing_snapshot() -> None:
    actual = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=True,
        strip_page_number_headers=False,
    )
    expected = json.loads(
        CURRENT_SNAPSHOT.read_text(encoding="utf-8")
    )

    assert actual == expected


def test_header_fix_snapshot_and_leaked_header_removal() -> None:
    fixed = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=True,
        strip_page_number_headers=True,
    )
    expected = json.loads(
        HEADER_FIX_SNAPSHOT.read_text(encoding="utf-8")
    )

    assert fixed == expected
    assert not any(
        HEADER_PATTERN.fullmatch(line)
        for line in _normalized_lines(fixed)
    )


def test_header_fix_does_not_change_body_phrase_occurrences() -> None:
    without_fix = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=True,
        strip_page_number_headers=False,
    )
    with_fix = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=True,
        strip_page_number_headers=True,
    )

    assert _body_phrase_count(without_fix) == 0
    assert _body_phrase_count(with_fix) == 0


def test_header_fix_preserves_page_count_and_primary_subset() -> None:
    current_fixed = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=True,
        strip_page_number_headers=True,
    )
    primary_fixed = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=False,
        strip_page_number_headers=True,
    )

    assert len(current_fixed) == len(primary_fixed)
    assert all(
        current.startswith(primary)
        for current, primary in zip(
            current_fixed,
            primary_fixed,
        )
    )


def test_header_fix_is_rejected_for_legacy_extraction() -> None:
    with pytest.raises(ValueError, match="legacy-pypdf"):
        PhaseOneRAG(
            PDF_PATH,
            extraction="legacy-pypdf",
            header_fix=True,
        )

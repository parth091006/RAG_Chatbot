import json
from pathlib import Path

from src.rag.ingestion.parser import (
    _is_repeated_header_or_footer,
    extract_pages_from_pdf,
)


PDF_PATH = Path("data/raw/self-attention-transformers-2023.pdf")
SNAPSHOT_PATH = Path("tests/fixtures/parser_current_snapshot.json")


def _normalized_page_text(text: str) -> str:
    return " ".join(text.casefold().split())


def test_default_parser_output_matches_snapshot() -> None:
    actual = extract_pages_from_pdf(PDF_PATH)
    expected = json.loads(
        SNAPSHOT_PATH.read_text(encoding="utf-8")
    )

    assert actual == expected


def test_primary_only_pages_are_prefixes_of_default_pages() -> None:
    current_pages = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=True,
    )
    primary_pages = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=False,
    )

    assert len(primary_pages) == len(current_pages)

    for current_page, primary_page in zip(
        current_pages,
        primary_pages,
    ):
        assert current_page.startswith(primary_page)


def test_primary_only_preserves_page_count() -> None:
    current_pages = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=True,
    )
    primary_pages = extract_pages_from_pdf(
        PDF_PATH,
        include_secondary=False,
    )

    assert len(primary_pages) == len(current_pages)


def test_repeated_headers_and_footers_are_absent_in_both_modes() -> None:
    assert _is_repeated_header_or_footer(
        "[draft] note 10:",
        page_height=1000.0,
        y0=20.0,
        y1=35.0,
    )
    assert _is_repeated_header_or_footer(
        "with deep learning",
        page_height=1000.0,
        y0=20.0,
        y1=35.0,
    )
    assert _is_repeated_header_or_footer(
        "7",
        page_height=1000.0,
        y0=950.0,
        y1=965.0,
    )

    for include_secondary in (True, False):
        pages = extract_pages_from_pdf(
            PDF_PATH,
            include_secondary=include_secondary,
        )
        normalized_text = [
            _normalized_page_text(page)
            for page in pages
        ]

        assert all(
            not any(
                line in {
                    "[draft] note 10 :",
                    "with deep learning",
                }
                for line in page.splitlines()
            )
            for page in normalized_text
        )

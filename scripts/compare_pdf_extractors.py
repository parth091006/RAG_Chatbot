from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader
import pymupdf


def extract_with_pypdf(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def extract_with_pymupdf(pdf_path: Path) -> list[str]:
    document = pymupdf.open(pdf_path)
    try:
        return [
            page.get_text("text")
            for page in document
        ]
    finally:
        document.close()


def print_preview(
    name: str,
    pages: list[str],
    max_pages: int = 5,
    max_chars: int = 2500,
) -> None:
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"Pages: {len(pages)}")

    for page_number, text in enumerate(
        pages[:max_pages],
        start=1,
    ):
        print()
        print("-" * 80)
        print(f"PAGE {page_number}")
        print("-" * 80)
        print(text[:max_chars])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/compare_pdf_extractors.py <pdf_path>"
        )

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pypdf_pages = extract_with_pypdf(pdf_path)
    pymupdf_pages = extract_with_pymupdf(pdf_path)

    print_preview(
        "PyPDF extraction",
        pypdf_pages,
    )

    print_preview(
        "PyMuPDF extraction",
        pymupdf_pages,
    )


if __name__ == "__main__":
    main()
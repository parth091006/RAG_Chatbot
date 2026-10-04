from __future__ import annotations

import argparse
import sys
from pathlib import Path


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract text from a PDF file using the installed PDF library."""
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        raise ImportError("pypdf is required to parse PDF documents.")

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument("pdf_path", help="Path to the PDF file to parse")
    args = parser.parse_args()

    try:
        text = extract_text_from_pdf(args.pdf_path)
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        print(text)
    except Exception as exc:  # pragma: no cover - command-line safety
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()

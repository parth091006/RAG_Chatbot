from __future__ import annotations

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

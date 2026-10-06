from __future__ import annotations

from pathlib import Path


def load_pdf_paths(directory: str | Path) -> list[Path]:
    """Return a sorted list of PDF files in a directory."""
    folder = Path(directory)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.pdf"))

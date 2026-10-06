import sys

try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except (AttributeError, ValueError):
    pass

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/inspect_pdf_blocks.py <pdf_path>"
        )

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    document = pymupdf.open(pdf_path)

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            print("=" * 100)
            print(f"PAGE {page_number}")
            print("=" * 100)

            blocks = page.get_text("blocks")

            for index, block in enumerate(blocks):
                x0, y0, x1, y1, text, *_ = block

                print()
                print(
                    f"BLOCK {index}: "
                    f"x0={x0:.1f}, "
                    f"y0={y0:.1f}, "
                    f"x1={x1:.1f}, "
                    f"y1={y1:.1f}"
                )

                print(repr(text[:500]))

    finally:
        document.close()


if __name__ == "__main__":
    main()
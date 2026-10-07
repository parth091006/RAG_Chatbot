from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, TypedDict


class RegionSpan(TypedDict):
    start: int
    end: int
    region: str
    header_like: bool


class RemovedBlock(TypedDict):
    text: str
    bbox: tuple[float, float, float, float]
    reason: str


class PageWithRegions(TypedDict):
    text: str
    region_spans: list[RegionSpan]
    removed_blocks: list[RemovedBlock]


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract layout-aware text from a PDF using PyMuPDF."""
    return "\n\n".join(extract_pages_from_pdf(file_path))


def extract_pages_from_pdf(
    file_path: str | Path,
    include_secondary: bool = True,
    strip_page_number_headers: bool = False,
) -> list[str]:
    """
    Extract page text using PyMuPDF block structure.

    The parser:
    - removes obvious repeated headers/footers
    - detects the primary text column from block geometry
    - keeps primary text in reading order
    - preserves secondary content such as diagrams and captions
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        raise ImportError(
            "pymupdf is required to parse PDF documents."
        )

    document = pymupdf.open(pdf_path)

    try:
        return [
            _extract_page_text(
                page,
                include_secondary=include_secondary,
                strip_page_number_headers=strip_page_number_headers,
            )
            for page in document
        ]
    finally:
        document.close()


def extract_pages_with_regions(
    file_path: str | Path,
    include_secondary: bool = True,
    strip_page_number_headers: bool = False,
) -> list[PageWithRegions]:
    """Extract page text with final block regions and removed-block metadata."""
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        raise ImportError(
            "pymupdf is required to parse PDF documents."
        )

    document = pymupdf.open(pdf_path)

    try:
        return [
            _extract_page_with_regions(
                page,
                include_secondary=include_secondary,
                strip_page_number_headers=strip_page_number_headers,
            )
            for page in document
        ]
    finally:
        document.close()


def _extract_page_text(
    page: Any,
    include_secondary: bool = True,
    strip_page_number_headers: bool = False,
) -> str:
    """Extract and order meaningful blocks from one PDF page."""
    return _extract_page_with_regions(
        page,
        include_secondary=include_secondary,
        strip_page_number_headers=strip_page_number_headers,
    )["text"]


def _extract_page_with_regions(
    page: Any,
    include_secondary: bool = True,
    strip_page_number_headers: bool = False,
) -> PageWithRegions:
    """Extract ordered blocks and retain their output provenance."""
    raw_blocks = page.get_text("blocks")

    blocks: list[dict[str, Any]] = []
    removed_blocks: list[RemovedBlock] = []

    for index, block in enumerate(raw_blocks):
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        if not text or not text.strip():
            continue

        cleaned_text = _clean_block_text(text)

        if not cleaned_text:
            continue

        if _is_repeated_header_or_footer(
            cleaned_text,
            page_height=page.rect.height,
            y0=y0,
            y1=y1,
            strip_page_number_headers=strip_page_number_headers,
        ):
            removed_blocks.append(
                {
                    "text": cleaned_text,
                    "bbox": (
                        float(x0),
                        float(y0),
                        float(x1),
                        float(y1),
                    ),
                    "reason": "repeated_header_or_footer",
                }
            )
            continue

        blocks.append(
            {
                "index": index,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "text": cleaned_text,
            }
        )

    if not blocks:
        return {
            "text": "",
            "region_spans": [],
            "removed_blocks": removed_blocks,
        }

    primary_blocks, secondary_blocks = _split_page_layout(
        blocks,
        page_width=float(page.rect.width),
    )

    primary_blocks.sort(
        key=lambda block: (
            block["y0"],
            block["x0"],
        )
    )

    secondary_blocks.sort(
        key=lambda block: (
            block["y0"],
            block["x0"],
        )
    )

    ordered_blocks = [
        (block, "primary")
        for block in primary_blocks
    ]

    if include_secondary:
        ordered_blocks += [
            (block, "secondary")
            for block in secondary_blocks
        ]

    page_text_parts: list[str] = []
    region_spans: list[RegionSpan] = []
    offset = 0

    for index, (block, region) in enumerate(ordered_blocks):
        if index:
            page_text_parts.append("\n\n")
            offset += 2

        block_text = block["text"]
        page_text_parts.append(block_text)
        region_spans.append(
            {
                "start": offset,
                "end": offset + len(block_text),
                "region": region,
                "header_like": _is_header_like_block(block_text),
            }
        )
        offset += len(block_text)

    return {
        "text": "".join(page_text_parts),
        "region_spans": region_spans,
        "removed_blocks": removed_blocks,
    }


def _split_page_layout(
    blocks: list[dict[str, Any]],
    page_width: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Separate the dominant/main text region from secondary content.

    The method uses block geometry rather than document-specific
    coordinates. Large text blocks are treated as stronger evidence
    of the primary text region than tiny diagram labels.
    """
    if len(blocks) <= 2:
        return blocks, []

    page_midpoint = page_width / 2

    left_blocks = [
        block
        for block in blocks
        if block["x0"] < page_midpoint
    ]

    right_blocks = [
        block
        for block in blocks
        if block["x0"] >= page_midpoint
    ]

    if not left_blocks or not right_blocks:
        return blocks, []

    left_score = _layout_region_score(left_blocks)
    right_score = _layout_region_score(right_blocks)

    if left_score >= right_score:
        return left_blocks, right_blocks

    return right_blocks, left_blocks


def _layout_region_score(
    blocks: list[dict[str, Any]],
) -> float:
    """
    Score a candidate text region.

    Larger blocks receive more weight because normal paragraphs
    generally occupy more area than diagram labels.
    """
    score = 0.0

    for block in blocks:
        width = max(block["x1"] - block["x0"], 0.0)
        height = max(block["y1"] - block["y0"], 0.0)
        text_length = len(block["text"])

        # Text density is a useful signal for prose blocks.
        density = text_length / max(width * height, 1.0)

        # Give substantial weight to text length and block width.
        score += (
            text_length
            + (width * 0.5)
            + (density * 100.0)
        )

    return score


def _clean_block_text(text: str) -> str:
    """Normalize whitespace inside an extracted text block."""
    text = text.replace("\x00", "")

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = [
        re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()
        for line in text.split("\n")
    ]

    lines = [
        line
        for line in lines
        if line
    ]

    return "\n".join(lines).strip()


def _is_repeated_header_or_footer(
    text: str,
    page_height: float,
    y0: float,
    y1: float,
    strip_page_number_headers: bool = False,
) -> bool:
    """Remove obvious repeated page-level header/footer content."""
    normalized = " ".join(
        text.lower().split()
    )

    if (
        y0 < page_height * 0.10
        and normalized.startswith(
            "[draft] note 10:"
        )
    ):
        return True

    if (
        y0 < page_height * 0.10
        and (
            normalized == "with deep learning"
            or (
                strip_page_number_headers
                and re.fullmatch(
                    r"with deep learning \d{1,3}",
                    normalized,
                )
                is not None
            )
        )
    ):
        return True

    if (
        len(normalized) <= 4
        and normalized.isdigit()
        and (
            y1 > page_height * 0.90
            or y0 < page_height * 0.10
        )
    ):
        return True

    return False


def _is_header_like_block(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    return re.fullmatch(
        r"with deep learning(?: \d{1,3})?",
        normalized,
    ) is not None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract layout-aware text from a PDF file."
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to parse",
    )

    args = parser.parse_args()

    try:
        text = extract_text_from_pdf(
            args.pdf_path
        )

        try:
            sys.stdout.reconfigure(
                encoding="utf-8",
                errors="replace",
            )
        except (AttributeError, ValueError):
            pass

        print(text)

    except Exception as exc:  # pragma: no cover
        parser.exit(
            status=1,
            message=f"Error: {exc}\n",
        )


if __name__ == "__main__":
    main()
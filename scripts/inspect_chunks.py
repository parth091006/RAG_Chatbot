from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.ingestion.parser import extract_pages_from_pdf
from src.rag.chunking.chunker import chunk_pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and inspect chunks from a PDF."
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Chunk size in words.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Chunk overlap in words.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First chunk number to display.",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=50,
        help="Last chunk number to display.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pages = extract_pages_from_pdf(pdf_path)

    document_id = f"doc_{pdf_path.stem}"

    chunks = chunk_pages(
        pages,
        document_id=document_id,
        document_name=pdf_path.name,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print("=" * 100)
    print("CHUNK INSPECTION")
    print("=" * 100)
    print(f"PDF: {pdf_path}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")
    print("=" * 100)

    start_index = max(args.start - 1, 0)
    end_index = min(args.end, len(chunks))

    for index in range(start_index, end_index):
        chunk = chunks[index]

        print()
        print("=" * 100)
        print(
            f"CHUNK {index + 1}/{len(chunks)}"
        )
        print("=" * 100)

        print(
            f"chunk_id: {chunk['chunk_id']}"
        )

        print(
            f"page_number: {chunk['page_number']}"
        )

        print(
            f"document_name: {chunk['document_name']}"
        )

        print("-" * 100)
        print(chunk["text"])
        print()


if __name__ == "__main__":
    main()
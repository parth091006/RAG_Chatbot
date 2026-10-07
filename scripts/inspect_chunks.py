from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag.evaluation.run_metadata import (
    build_run_metadata,
    resolve_extraction,
    safe_write_json,
)
from src.rag.pipeline import PhaseOneRAG


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

    parser.add_argument(
        "--extraction",
        choices=["legacy-pypdf", "current", "primary-only"],
        default=None,
    )

    parser.add_argument(
        "--header-fix",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for inspected chunks.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    try:
        extraction, metadata_enabled = resolve_extraction(
            args.extraction,
            args.header_fix,
            "current",
        )
    except ValueError as exc:
        parser.error(str(exc))

    rag = PhaseOneRAG(
        pdf_path,
        extraction=extraction,
        header_fix=args.header_fix,
    )
    chunks = rag.pdf_to_chunks(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    if args.output is not None:
        serialized_chunks = []
        for chunk in chunks:
            serialized = {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "chunk_text_hash": chunk["chunk_text_hash"],
                "char_count": len(str(chunk["text"])),
                "text": chunk["text"],
            }
            if metadata_enabled:
                serialized["secondary_char_ratio"] = chunk.get(
                    "secondary_char_ratio"
                )
            serialized_chunks.append(serialized)

        metadata = None
        if metadata_enabled:
            metadata = build_run_metadata(
                extraction=extraction,
                header_fix=args.header_fix,
                embedding_method="none",
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                pipeline=rag,
                pdf_path=pdf_path,
            )

        safe_write_json(
            args.output,
            serialized_chunks,
            metadata=metadata,
            force=args.force,
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
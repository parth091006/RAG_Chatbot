from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag.pipeline import PhaseOneRAG
from src.rag.evaluation.retrieval_metrics import recall_at_k


def estimated_tokens(text: str) -> int:
    return len(
        re.findall(
            r"\w+|[^\w\s]",
            text,
        )
    )


def run_strategy(
    name: str,
    pdf_path: Path,
    questions: list[dict],
    chunk_size: int,
    overlap: int,
    strategy: str,
) -> dict:
    rag = PhaseOneRAG(pdf_path)

    chunks = rag.pdf_to_chunks(
        chunk_size=chunk_size,
        overlap=overlap,
        strategy=strategy,
    )

    rag.chunks_to_embeddings(chunks)

    recalls = []

    for question in questions:
        results = rag.question_to_scored_chunks(
            question["question"],
            top_k=5,
        )

        relevant_ids = set(
            question.get("relevant_chunks", [])
        )

        if relevant_ids:
            retrieved_ids = [
                result["chunk"]["chunk_id"]
                for result in results
            ]

            recalls.append(
                recall_at_k(
                    relevant_ids,
                    retrieved_ids,
                    5,
                )
            )

    total_words = sum(
        len(str(chunk["text"]).split())
        for chunk in chunks
    )

    total_tokens = sum(
        estimated_tokens(str(chunk["text"]))
        for chunk in chunks
    )

    return {
        "name": name,
        "strategy": strategy,
        "chunk_size_words": chunk_size,
        "overlap_words": overlap,
        "chunk_count": len(chunks),
        "average_chunk_size_words": round(
            total_words / max(len(chunks), 1),
            2,
        ),
        "average_chunk_size_estimated_tokens": round(
            total_tokens / max(len(chunks), 1),
            2,
        ),
        "recall_at_5": (
            round(
                sum(recalls) / len(recalls),
                4,
            )
            if recalls
            else None
        ),
        "recall_basis": (
            "hand-labeled relevant chunk IDs"
            if recalls
            else "not available; annotate relevant_chunks"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare chunking strategies "
            "for the RAG PDF."
        )
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            "data/raw/"
            "self-attention-transformers-2023.pdf"
        ),
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(
            "data/evaluation/"
            "retrieval_questions.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/evaluation/"
            "chunking_experiment_results.json"
        ),
    )

    args = parser.parse_args()

    questions = json.loads(
        args.questions.read_text(
            encoding="utf-8"
        )
    )

    experiments = [
        run_strategy(
            "A - Current baseline",
            args.pdf,
            questions,
            250,
            50,
            "fixed",
        ),
        run_strategy(
            "B - Smaller chunks",
            args.pdf,
            questions,
            150,
            30,
            "fixed",
        ),
        run_strategy(
            "C - Section-aware chunks",
            args.pdf,
            questions,
            250,
            50,
            "section",
        ),
    ]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            experiments,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Name                         "
        "Chunks  Avg words  Avg tokens  Recall@5"
    )
    print("-" * 74)

    for result in experiments:
        recall = (
            "null"
            if result["recall_at_5"] is None
            else f"{result['recall_at_5']:.4f}"
        )

        print(
            f"{result['name']:<28}"
            f" {result['chunk_count']:>6}"
            f" {result['average_chunk_size_words']:>10.2f}"
            f" {result['average_chunk_size_estimated_tokens']:>11.2f}"
            f" {recall:>9}"
        )

    print(
        f"\nSaved results to {args.output}"
    )

    print(
        "Recall@5 is shown only after "
        "relevant_chunks are hand-labeled."
    )


if __name__ == "__main__":
    main()
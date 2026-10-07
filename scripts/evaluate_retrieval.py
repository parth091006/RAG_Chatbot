from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag.pipeline import PhaseOneRAG
from src.rag.evaluation.run_metadata import (
    build_run_metadata,
    resolve_extraction,
    safe_write_json,
    serialize_chunk,
)
from src.rag.evaluation.retrieval_metrics import (
    mrr,
    recall_at_k,
)


def evaluate_questions(
    pdf_path: Path,
    questions_path: Path,
    output_path: Path,
    top_k: int = 5,
    embedding_method: str = "bow",
    extraction: str = "legacy-pypdf",
    header_fix: bool = False,
    metadata_enabled: bool = False,
    force: bool = False,
) -> None:
    questions = json.loads(
        questions_path.read_text(
            encoding="utf-8"
        )
    )

    rag = PhaseOneRAG(
        pdf_path,
        embedding_method=embedding_method,
        extraction=extraction,
        header_fix=header_fix,
    )

    rag.pdf_to_chunks()
    rag.chunks_to_embeddings()

    results = []

    for item in questions:
        question = item["question"]

        scored_results = rag.question_to_scored_chunks(
            question,
            top_k=top_k,
        )

        retrieved_ids = [
            result["chunk"]["chunk_id"]
            for result in scored_results
        ]

        relevant_ids = set(
            item.get("relevant_chunks", [])
        )

        serialized_results = [
            {
                "chunk": serialize_chunk(
                    result["chunk"],
                    include_optional=metadata_enabled,
                ),
                "score": result["score"],
            }
            for result in scored_results
        ]

        results.append(
            {
                "id": item.get("id"),
                "question": question,
                "results": serialized_results,
                "metrics": {
                    "recall_at_1": (
                        recall_at_k(
                            relevant_ids,
                            retrieved_ids,
                            1,
                        )
                        if relevant_ids
                        else None
                    ),
                    "recall_at_3": (
                        recall_at_k(
                            relevant_ids,
                            retrieved_ids,
                            3,
                        )
                        if relevant_ids
                        else None
                    ),
                    "recall_at_5": (
                        recall_at_k(
                            relevant_ids,
                            retrieved_ids,
                            5,
                        )
                        if relevant_ids
                        else None
                    ),
                    "mrr": (
                        mrr(
                            relevant_ids,
                            retrieved_ids,
                        )
                        if relevant_ids
                        else None
                    ),
                },
            }
        )

    metadata = None
    if metadata_enabled:
        metadata = build_run_metadata(
            extraction=extraction,
            header_fix=header_fix,
            embedding_method=embedding_method,
            chunk_size=250,
            overlap=50,
            pipeline=rag,
            pdf_path=pdf_path,
            question_path=questions_path,
            top_k=top_k,
        )

    safe_write_json(
        output_path,
        {
            "embedding_method": embedding_method,
            "top_k": top_k,
            "results": results,
        },
        metadata=metadata,
        force=force,
    )

    print(
        f"\nEmbedding method: {embedding_method}"
    )

    for result in results:
        print(
            f"\n{result['id']}. "
            f"{result['question']}"
        )

        metrics = result["metrics"]

        print(
            f"  Recall@1={metrics['recall_at_1']} "
            f"Recall@3={metrics['recall_at_3']} "
            f"Recall@5={metrics['recall_at_5']} "
            f"MRR={metrics['mrr']}"
        )

        hits = result["results"]

        for rank, hit in enumerate(hits, start=1):
            chunk = hit["chunk"]

            print(
                f"  {rank}. "
                f"{chunk['chunk_id']} "
                f"score={hit['score']:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval for a JSON question set."
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
            "retrieval_results.json"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--embedding-method",
        choices=["bow", "tfidf", "semantic"],
        default="bow",
        help=(
            "Embedding method to evaluate: "
            "bow, tfidf, or semantic."
        ),
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

    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )
    except (AttributeError, ValueError):
        pass

    try:
        extraction, metadata_enabled = resolve_extraction(
            args.extraction,
            args.header_fix,
            "legacy-pypdf",
        )
    except ValueError as exc:
        parser.error(str(exc))

    evaluate_questions(
        args.pdf,
        args.questions,
        args.output,
        args.top_k,
        args.embedding_method,
        extraction,
        args.header_fix,
        metadata_enabled,
        args.force,
    )


if __name__ == "__main__":
    main()
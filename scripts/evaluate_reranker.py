from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag.pipeline import PhaseOneRAG
from src.rag.retrieval.reranker import Reranker
from src.rag.evaluation.retrieval_metrics import mrr, recall_at_k


def evaluate_questions(
    pdf_path: Path,
    questions_path: Path,
    output_path: Path,
    candidate_k: int = 10,
    top_k: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> None:
    """Evaluate TF-IDF retrieval followed by cross-encoder reranking."""

    questions = json.loads(
        questions_path.read_text(encoding="utf-8")
    )

    # TF-IDF is the current Phase-1 retrieval winner.
    rag = PhaseOneRAG(
        pdf_path,
        embedding_method="tfidf",
    )

    # Keep the canonical Phase-1 chunking configuration.
    chunks = rag.pdf_to_chunks(
        strategy="fixed",
        chunk_size=250,
        overlap=50,
    )

    rag.chunks_to_embeddings(chunks)

    reranker = Reranker(
        model_name=reranker_model,
        top_k=top_k,
    )

    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        relevant_ids = set(
            item.get("relevant_chunks", [])
        )

        # Stage 1:
        # TF-IDF retrieves a larger candidate pool.
        retrieval_results = rag.question_to_scored_chunks(
            question,
            top_k=candidate_k,
        )

        candidate_documents = [
            (
                index,
                chunks[index]["text"],
            )
            for index, _ in [
                (
                    chunks.index(result["chunk"]),
                    result["score"],
                )
                for result in retrieval_results
            ]
        ]

        # Stage 2:
        # Cross-encoder reranks the candidate pool.
        reranked_results = reranker.rerank(
            query=question,
            candidates=candidate_documents,
        )

        retrieved_ids = [
            chunks[index]["chunk_id"]
            for index, _ in reranked_results
        ]

        question_result = {
            "id": question_id,
            "question": question,
            "candidate_k": candidate_k,
            "results": [],
            "metrics": {
                "recall_at_1": recall_at_k(
                    relevant_ids,
                    retrieved_ids,
                    1,
                ),
                "recall_at_3": recall_at_k(
                    relevant_ids,
                    retrieved_ids,
                    3,
                ),
                "recall_at_5": recall_at_k(
                    relevant_ids,
                    retrieved_ids,
                    5,
                ),
                "mrr": mrr(
                    relevant_ids,
                    retrieved_ids,
                ),
            },
        }

        for index, score in reranked_results:
            question_result["results"].append(
                {
                    "chunk": chunks[index],
                    "score": float(score),
                }
            )

        results.append(question_result)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Print per-question results.
    for result in results:
        metrics = result["metrics"]

        print(f"\n{result['id']}: {result['question']}")
        print(
            f"  R@1={metrics['recall_at_1']:.4f} "
            f"R@3={metrics['recall_at_3']:.4f} "
            f"R@5={metrics['recall_at_5']:.4f} "
            f"MRR={metrics['mrr']:.4f}"
        )

        for rank, hit in enumerate(
            result["results"],
            start=1,
        ):
            chunk = hit["chunk"]

            print(
                f"  {rank}. "
                f"{chunk['chunk_id']} "
                f"score={hit['score']:.6f}"
            )

    # Calculate overall averages.
    count = len(results)

    if count:
        avg_r1 = sum(
            result["metrics"]["recall_at_1"]
            for result in results
        ) / count

        avg_r3 = sum(
            result["metrics"]["recall_at_3"]
            for result in results
        ) / count

        avg_r5 = sum(
            result["metrics"]["recall_at_5"]
            for result in results
        ) / count

        avg_mrr = sum(
            result["metrics"]["mrr"]
            for result in results
        ) / count

        print("\n" + "=" * 60)
        print("TF-IDF + CROSS-ENCODER RERANKER SUMMARY")
        print("=" * 60)
        print(f"Candidate K : {candidate_k}")
        print(f"Final K     : {top_k}")
        print(f"Reranker    : {reranker_model}")
        print(f"Average R@1 : {avg_r1:.4f}")
        print(f"Average R@3 : {avg_r3:.4f}")
        print(f"Average R@5 : {avg_r5:.4f}")
        print(f"Average MRR : {avg_mrr:.4f}")

    print(
        f"\nSaved results to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TF-IDF retrieval followed by "
            "cross-encoder reranking."
        )
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "self-attention-transformers-2023.pdf"
        ),
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "evaluation"
            / "retrieval_questions.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "evaluation"
            / "reranker_results.json"
        ),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of candidates retrieved before reranking.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates kept after reranking.",
    )

    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    args = parser.parse_args()

    if args.candidate_k <= 0:
        parser.error("--candidate-k must be greater than 0.")

    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0.")

    if args.top_k > args.candidate_k:
        parser.error(
            "--top-k cannot be greater than --candidate-k."
        )

    evaluate_questions(
        pdf_path=args.pdf,
        questions_path=args.questions,
        output_path=args.output,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        reranker_model=args.reranker_model,
    )


if __name__ == "__main__":
    main()
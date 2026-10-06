from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag.pipeline import PhaseOneRAG
from src.rag.retrieval.hybrid import HybridRetriever
from src.rag.evaluation.retrieval_metrics import mrr, recall_at_k


def evaluate_questions(
    pdf_path: Path,
    questions_path: Path,
    output_path: Path,
    top_k: int = 5,
) -> None:
    """Evaluate hybrid BM25 + semantic retrieval using RRF."""

    questions = json.loads(
        questions_path.read_text(encoding="utf-8")
    )

    # Use semantic embeddings for the dense side of the hybrid retriever.
    rag = PhaseOneRAG(
        pdf_path,
        embedding_method="semantic",
    )

    # Keep the canonical Phase-1 chunking configuration:
    # 250 words with 50-word overlap.
    chunks = rag.pdf_to_chunks(
        strategy="fixed",
        chunk_size=250,
        overlap=50,
    )

    embeddings = rag.chunks_to_embeddings(chunks)

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    hybrid = HybridRetriever(
        documents=documents,
        embeddings=embeddings,
    )

    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        relevant_ids = set(
            item.get("relevant_chunks", [])
        )

        # Generate the semantic query embedding.
        query_embedding = rag.embedder.embed(question)

        hybrid_hits = hybrid.search(
            query=question,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        retrieved_ids = [
            chunks[index]["chunk_id"]
            for index, _ in hybrid_hits
        ]

        question_result = {
            "id": question_id,
            "question": question,
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

        for index, score in hybrid_hits:
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
        print("HYBRID RETRIEVAL SUMMARY")
        print("=" * 60)
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
            "Evaluate hybrid BM25 + semantic retrieval "
            "using Reciprocal Rank Fusion."
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
            / "hybrid_retrieval_results.json"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    evaluate_questions(
        pdf_path=args.pdf,
        questions_path=args.questions,
        output_path=args.output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
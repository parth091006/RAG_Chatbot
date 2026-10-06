from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag.evaluation.generation_metrics import (
    answer_relevance_score,
    faithfulness_score,
)
from src.rag.pipeline import PhaseOneRAG


def evaluate_questions(
    pdf_path: Path,
    questions_path: Path,
    output_path: Path,
    top_k: int = 5,
) -> None:
    """Evaluate deterministic generation using retrieved contexts."""

    questions = json.loads(
        questions_path.read_text(encoding="utf-8")
    )

    # Use the current Phase-1 retrieval winner.
    rag = PhaseOneRAG(
        pdf_path,
        embedding_method="tfidf",
    )

    # Canonical Phase-1 chunking configuration.
    chunks = rag.pdf_to_chunks(
        strategy="fixed",
        chunk_size=250,
        overlap=50,
    )

    rag.chunks_to_embeddings(chunks)

    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        # Retrieve the current top-k contexts.
        contexts = rag.question_to_top_k_chunks(
            question,
            top_k=top_k,
        )

        # Convert chunks into the context format expected by Generator.
        generator_contexts = [
            {
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
            }
            for chunk in contexts
        ]

        generated = rag.generator.generate(
            question,
            generator_contexts,
        )

        evidence = "\n\n".join(
            str(context["text"])
            for context in generator_contexts
        )

        answer = generated["answer"]

        result = {
            "id": question_id,
            "question": question,
            "answer": answer,
            "sources": generated["sources"],
            "metrics": {
                "faithfulness": faithfulness_score(
                    answer,
                    evidence,
                ),
                "answer_relevance": answer_relevance_score(
                    answer,
                    question,
                ),
            },
        }

        results.append(result)

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

    # Print individual results.
    for result in results:
        metrics = result["metrics"]

        print(f"\n{result['id']}: {result['question']}")
        print(f"  Answer: {result['answer']}")
        print(
            f"  Faithfulness proxy: "
            f"{metrics['faithfulness']:.4f}"
        )
        print(
            f"  Answer relevance proxy: "
            f"{metrics['answer_relevance']:.4f}"
        )

        print("  Sources:")

        for source in result["sources"]:
            print(
                f"    {source['citation_id']} "
                f"{source['document']} "
                f"page={source['page']} "
                f"chunk={source['chunk_id']}"
            )

    # Calculate overall averages.
    count = len(results)

    if count:
        avg_faithfulness = sum(
            result["metrics"]["faithfulness"]
            for result in results
        ) / count

        avg_relevance = sum(
            result["metrics"]["answer_relevance"]
            for result in results
        ) / count

        print("\n" + "=" * 60)
        print("GENERATION EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Top K contexts       : {top_k}")
        print(
            f"Average faithfulness: "
            f"{avg_faithfulness:.4f}"
        )
        print(
            f"Average relevance   : "
            f"{avg_relevance:.4f}"
        )

    print(
        f"\nSaved results to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Phase-1 deterministic "
            "extractive generator."
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
            / "generation_results.json"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved contexts used for generation.",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0.")

    evaluate_questions(
        pdf_path=args.pdf,
        questions_path=args.questions,
        output_path=args.output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
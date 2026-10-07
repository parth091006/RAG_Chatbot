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
from src.rag.evaluation.run_metadata import (
    build_run_metadata,
    resolve_extraction,
    safe_write_json,
)
from src.rag.evaluation.noise_metrics import (
    answer_evidence_support,
    answer_header_leak,
    answer_secondary_ratio,
    answer_unlocatable_sentence_count,
    context_noise_per_chunk,
    diagram_noise_ratio,
    header_leak_counts,
    document_label_like_stats,
    label_like_char_share,
)
from src.rag.pipeline import PhaseOneRAG
from src.rag.retrieval.reranker import Reranker


def evaluate_questions(
    pdf_path: Path,
    questions_path: Path,
    output_path: Path,
    candidate_k: int = 10,
    top_k: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    extraction: str = "legacy-pypdf",
    header_fix: bool = False,
    metadata_enabled: bool = False,
    force: bool = False,
) -> None:
    """Evaluate generation using TF-IDF retrieval followed by reranking."""

    questions = json.loads(
        questions_path.read_text(encoding="utf-8")
    )

    # Current Phase-1 retrieval winner.
    rag = PhaseOneRAG(
        pdf_path,
        embedding_method="tfidf",
        extraction=extraction,
        header_fix=header_fix,
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

    label_stats = document_label_like_stats(chunks)

    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]

        # ---------------------------------------------------------
        # Stage 1: TF-IDF retrieves a larger candidate pool.
        # ---------------------------------------------------------
        retrieval_results = rag.question_to_scored_chunks(
            question,
            top_k=candidate_k,
        )

        candidate_documents = [
            (
                chunks.index(result["chunk"]),
                str(result["chunk"]["text"]),
            )
            for result in retrieval_results
        ]

        # ---------------------------------------------------------
        # Stage 2: Cross-encoder reranks the candidate pool.
        # ---------------------------------------------------------
        reranked_results = reranker.rerank(
            query=question,
            candidates=candidate_documents,
        )

        # ---------------------------------------------------------
        # Stage 3: Convert reranked chunks into Generator contexts.
        # ---------------------------------------------------------
        generator_contexts = [
            {
                "chunk_id": chunks[index]["chunk_id"],
                "document_name": chunks[index]["document_name"],
                "page_number": chunks[index]["page_number"],
                "text": chunks[index]["text"],
            }
            for index, _ in reranked_results
        ]

        # ---------------------------------------------------------
        # Stage 4: Generate the answer.
        # ---------------------------------------------------------
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
            "candidate_k": candidate_k,
            "top_k": top_k,
            "answer": answer,
            "sources": generated["sources"],
            "retrieved_chunks": [
                {
                    "chunk_id": chunks[index]["chunk_id"],
                    "reranker_score": float(score),
                }
                for index, score in reranked_results
            ],
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

        if metadata_enabled:
            context_chunks = [
                chunks[index]
                for index, _ in reranked_results
            ]
            result["noise_diagnostics"] = {
                "header_leak": header_leak_counts(
                    context_chunks
                ),
                "diagram_noise_ratio": diagram_noise_ratio(
                    context_chunks
                ),
                "context_noise_per_chunk": context_noise_per_chunk(
                    context_chunks
                ),
                "answer_secondary_ratio": answer_secondary_ratio(
                    answer,
                    context_chunks,
                ),
                "answer_header_leak": answer_header_leak(
                    answer,
                    context_chunks,
                ),
                "answer_unlocatable_sentence_count": (
                    answer_unlocatable_sentence_count(
                        answer,
                        context_chunks,
                    )
                ),
                "answer_evidence_support": answer_evidence_support(
                    answer,
                    context_chunks,
                ),
                "label_like_char_share": label_like_char_share(
                    answer,
                    context_chunks,
                ),
                "label_like_threshold": label_stats["threshold"],
                "label_like_document_stats": label_stats,
            }

        results.append(result)

    metadata = None
    if metadata_enabled:
        metadata = build_run_metadata(
            extraction=extraction,
            header_fix=header_fix,
            embedding_method="tfidf",
            chunk_size=250,
            overlap=50,
            pipeline=rag,
            pdf_path=pdf_path,
            question_path=questions_path,
            candidate_k=candidate_k,
            top_k=top_k,
            reranker_model=reranker_model,
        )

    safe_write_json(
        output_path,
        results,
        metadata=metadata,
        force=force,
    )

    # -------------------------------------------------------------
    # Print individual results.
    # -------------------------------------------------------------
    for result in results:
        metrics = result["metrics"]

        print(
            f"\n{result['id']}: "
            f"{result['question']}"
        )

        print(
            f"  Answer: {result['answer']}"
        )

        print(
            f"  Faithfulness proxy: "
            f"{metrics['faithfulness']:.4f}"
        )

        print(
            f"  Answer relevance proxy: "
            f"{metrics['answer_relevance']:.4f}"
        )

        print("  Reranked sources:")

        for rank, source in enumerate(
            result["sources"],
            start=1,
        ):
            print(
                f"    {rank}. "
                f"{source['citation_id']} "
                f"{source['document']} "
                f"page={source['page']} "
                f"chunk={source['chunk_id']}"
            )

    # -------------------------------------------------------------
    # Calculate overall averages.
    # -------------------------------------------------------------
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
        print("RERANKED GENERATION EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Candidate K          : {candidate_k}")
        print(f"Final K              : {top_k}")
        print(f"Reranker             : {reranker_model}")
        print(
            f"Average faithfulness: "
            f"{avg_faithfulness:.4f}"
        )
        print(
            f"Average relevance   : "
            f"{avg_relevance:.4f}"
        )

    if metadata_enabled and results:
        diagnostics = [
            result["noise_diagnostics"]
            for result in results
        ]
        answer_ratios = [
            item["answer_secondary_ratio"]
            for item in diagnostics
            if item["answer_secondary_ratio"] is not None
        ]
        context_means = [
            item["context_noise_per_chunk"]["mean"]
            for item in diagnostics
            if item["context_noise_per_chunk"]["mean"] is not None
        ]
        evidence_support = [
            item["answer_evidence_support"]
            for item in diagnostics
        ]
        label_shares = [
            item["label_like_char_share"]
            for item in diagnostics
        ]
        print("\n" + "=" * 60)
        print("NOISE DIAGNOSTICS SUMMARY")
        print("=" * 60)
        print(
            "Average diagram noise ratio: "
            f"{sum(item['diagram_noise_ratio'] for item in diagnostics if item['diagram_noise_ratio'] is not None) / max(sum(item['diagram_noise_ratio'] is not None for item in diagnostics), 1):.4f}"
        )
        print(
            "Average context secondary ratio: "
            f"{sum(context_means) / len(context_means) if context_means else None}"
        )
        print(
            "Average answer secondary ratio: "
            f"{sum(answer_ratios) / len(answer_ratios) if answer_ratios else None}"
        )
        print(
            "Average answer evidence support: "
            f"{sum(evidence_support) / len(evidence_support):.4f}"
        )
        print(
            "Average label-like character share: "
            f"{sum(label_shares) / len(label_shares):.4f}"
        )

    print(
        f"\nSaved results to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generation using TF-IDF retrieval, "
            "cross-encoder reranking, and the Phase-1 generator."
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
            / "reranked_generation_results.json"
        ),
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help=(
            "Number of TF-IDF candidates retrieved "
            "before reranking."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of contexts retained after "
            "cross-encoder reranking."
        ),
    )

    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
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

    if args.candidate_k <= 0:
        parser.error(
            "--candidate-k must be greater than 0."
        )

    if args.top_k <= 0:
        parser.error(
            "--top-k must be greater than 0."
        )

    if args.top_k > args.candidate_k:
        parser.error(
            "--top-k cannot be greater than --candidate-k."
        )

    try:
        extraction, metadata_enabled = resolve_extraction(
            args.extraction,
            args.header_fix,
            "legacy-pypdf",
        )
    except ValueError as exc:
        parser.error(str(exc))

    evaluate_questions(
        pdf_path=args.pdf,
        questions_path=args.questions,
        output_path=args.output,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        reranker_model=args.reranker_model,
        extraction=extraction,
        header_fix=args.header_fix,
        metadata_enabled=metadata_enabled,
        force=args.force,
    )


if __name__ == "__main__":
    main()
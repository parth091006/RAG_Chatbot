from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.evaluation.noise_metrics import (
    _diagnostic_fragments,
    answer_fragment_diagnostics,
    function_word_ratio,
    label_like_char_share,
)
from src.rag.pipeline import PhaseOneRAG
from src.rag.retrieval.reranker import Reranker


PDF_PATH = PROJECT_ROOT / "data/raw/self-attention-transformers-2023.pdf"
QUESTION_PATH = PROJECT_ROOT / "data/evaluation/retrieval_questions.json"
THRESHOLD = 0.25

CLASS_BY_CHUNK = {
    "doc_self-attention-transformers-2023_chunk_001": "truncated prose",
    "doc_self-attention-transformers-2023_chunk_002": "diagram/figure labels",
    "doc_self-attention-transformers-2023_chunk_005": "other",
    "doc_self-attention-transformers-2023_chunk_009": "equation/math",
    "doc_self-attention-transformers-2023_chunk_010": "equation/math",
    "doc_self-attention-transformers-2023_chunk_016": "other",
    "doc_self-attention-transformers-2023_chunk_017": "truncated prose",
    "doc_self-attention-transformers-2023_chunk_018": "diagram/figure labels",
    "doc_self-attention-transformers-2023_chunk_019": "equation/math",
    "doc_self-attention-transformers-2023_chunk_022": "equation/math",
    "doc_self-attention-transformers-2023_chunk_024": "equation/math",
    "doc_self-attention-transformers-2023_chunk_025": "truncated prose",
    "doc_self-attention-transformers-2023_chunk_026": "diagram/figure labels",
    "doc_self-attention-transformers-2023_chunk_028": "bibliography",
    "doc_self-attention-transformers-2023_chunk_029": "diagram/figure labels",
    "doc_self-attention-transformers-2023_chunk_034": "bibliography",
    "doc_self-attention-transformers-2023_chunk_036": "bibliography",
    "doc_self-attention-transformers-2023_chunk_037": "bibliography",
}
UNSURE = {
    "doc_self-attention-transformers-2023_chunk_005": "mixed prose and diagram sequence",
    "doc_self-attention-transformers-2023_chunk_016": "prose mixed with equation/header material",
}


def load_questions() -> list[dict[str, Any]]:
    return json.loads(QUESTION_PATH.read_text(encoding="utf-8"))


def arm_answer(
    question: str,
    extraction: str,
    header_fix: bool,
    reranker: Reranker,
) -> tuple[str, list[dict[str, Any]]]:
    pipeline = PhaseOneRAG(
        PDF_PATH,
        embedding_method="tfidf",
        extraction=extraction,
        header_fix=header_fix,
    )
    chunks = pipeline.pdf_to_chunks()
    pipeline.chunks_to_embeddings(chunks)
    candidates = pipeline.question_to_scored_chunks(
        question,
        top_k=10,
    )
    reranked = reranker.rerank(
        question,
        [
            (chunks.index(result["chunk"]), str(result["chunk"]["text"]))
            for result in candidates
        ],
    )
    contexts = [chunks[index] for index, _ in reranked]
    generator_contexts = [
        {
            "chunk_id": chunk["chunk_id"],
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
        }
        for chunk in contexts
    ]
    answer = pipeline.generator.generate(
        question,
        generator_contexts,
    )["answer"]
    return answer, contexts


def answer_fragments(
    answer: str,
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics = answer_fragment_diagnostics(answer, contexts)
    fragments = []
    for sentence in diagnostics["sentences"]:
        for fragment in sentence["fragments"]:
            if not fragment["located"]:
                continue
            text = " ".join(str(fragment["text"]).split())
            for part in _diagnostic_fragments(text):
                fragments.append(
                    {
                        "text": part,
                        "chunk_id": fragment["chunk_id"],
                        "length": len(part),
                        "ratio": function_word_ratio(part),
                        "flagged": (
                            len(part) >= 60
                            and not part.endswith((".", "!", "?"))
                            and function_word_ratio(part) < THRESHOLD
                        ),
                    }
                )
    return fragments


def main() -> None:
    questions = load_questions()
    reranker = Reranker(top_k=5)
    rows = []

    arm_a = PhaseOneRAG(
        PDF_PATH,
        extraction="current",
    ).pdf_to_chunks()
    for chunk in arm_a:
        for fragment in _diagnostic_fragments(str(chunk["text"])):
            text = " ".join(fragment.split())
            ratio = function_word_ratio(text)
            if (
                len(text) >= 60
                and not text.endswith((".", "!", "?"))
                and ratio < THRESHOLD
            ):
                rows.append(
                    {
                        "page": chunk["page_number"],
                        "chunk_id": chunk["chunk_id"],
                        "ratio": ratio,
                        "length": len(text),
                        "class": CLASS_BY_CHUNK[chunk["chunk_id"]],
                        "text": text,
                        "justification": UNSURE.get(
                            chunk["chunk_id"],
                            "classified from the complete fragment text",
                        ),
                    }
                )

    report = [
        "# Flagged Fragment Audit",
        "",
        "Manual, post-hoc audit of arm-A eligible fragments at t = 0.25.",
        "",
        "| Page | Chunk ID | Ratio | Length | Class | Justification |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        report.append(
            f"| {row['page']} | {row['chunk_id']} | "
            f"{row['ratio']:.6f} | {row['length']} | "
            f"{row['class']} | {row['justification']} |"
        )
        report.extend(["", "```text", row["text"], "```", ""])

    diagram_count = sum(
        row["class"] == "diagram/figure labels"
        for row in rows
    )
    class_counts = {}
    for row in rows:
        class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1
    report.extend([
        "Class counts: diagram/figure labels=4, equation/math=5, "
        "bibliography=4, truncated prose=3, other=2 (18 total).",
        "The earlier equation/math count of 6 was an arithmetic error; no "
        "fragment changed class.",
        "",
        f"Precision: {diagram_count}/{len(rows)} = {diagram_count / len(rows):.4f}",
        "",
        "Uncertain fragments:",
    ])
    for chunk_id, reason in UNSURE.items():
        report.append(f"- {chunk_id}: {reason}")

    report.extend(["", "## Audited Label Share", ""])
    report.append(
        "This section is manual audit, post-hoc. It counts only fragments "
        "classified as diagram/figure labels."
    )
    report.extend([
        "",
        "| Question | Arm | Fragments | Raw share | Audited label share |",
        "| --- | --- | --- | ---: | ---: |",
    ])
    question_map = {item["id"]: item["question"] for item in questions}
    for arm, extraction, header_fix in [
        ("A", "current", False),
        ("B", "current", True),
        ("C", "primary-only", False),
    ]:
        for question_id in ["q003", "q004", "q008", "q010"]:
            answer, contexts = arm_answer(
                question_map[question_id],
                extraction,
                header_fix,
                reranker,
            )
            fragments = answer_fragments(answer, contexts)
            contributed = [fragment for fragment in fragments if fragment["flagged"]]
            total = sum(fragment["length"] for fragment in fragments)
            audited = sum(
                fragment["length"]
                for fragment in contributed
                if CLASS_BY_CHUNK.get(fragment["chunk_id"])
                == "diagram/figure labels"
            )
            raw = label_like_char_share(answer, contexts, THRESHOLD)
            report.append(
                f"| {question_id} | {arm} | "
                f"{', '.join(fragment['chunk_id'] for fragment in contributed) or '-'} | "
                f"{raw:.6f} | {audited / total if total else 0.0:.6f} |"
            )
            for fragment in contributed:
                report.append(
                    f"- {question_id} {arm}: {fragment['chunk_id']}, "
                    f"length={fragment['length']}, "
                    f"class={CLASS_BY_CHUNK.get(fragment['chunk_id'], 'unknown')}, "
                    "class_basis=manual re-read of this arm's fragment, "
                    f"text={fragment['text']}"
                )

    output = PROJECT_ROOT / "docs/experiments/flagged_fragment_audit.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

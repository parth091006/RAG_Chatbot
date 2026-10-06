from __future__ import annotations

import re


def _normalize_terms(text: str) -> set[str]:
    """Convert text into normalized lexical terms."""

    return {
        term.lower()
        for term in re.findall(
            r"\b[\w'-]+\b",
            text,
        )
        if len(term) > 2
    }


def faithfulness_score(
    answer: str,
    evidence: str,
) -> float:
    """Estimate how much of the answer is supported by the evidence.

    This is a lexical proxy, not a semantic faithfulness metric.

    A score of 1.0 means every normalized content term in the answer
    also appears in the supplied evidence.
    """

    if not answer or not evidence:
        return 0.0

    answer_terms = _normalize_terms(answer)
    evidence_terms = _normalize_terms(evidence)

    if not answer_terms:
        return 0.0

    supported_terms = answer_terms & evidence_terms

    return round(
        len(supported_terms) / len(answer_terms),
        3,
    )


def answer_relevance_score(
    answer: str,
    question: str,
) -> float:
    """Estimate lexical overlap between the answer and question.

    This is a lexical proxy, not a semantic relevance metric.

    A score of 1.0 means all normalized content terms from the question
    also appear in the answer.
    """

    if not answer or not question:
        return 0.0

    question_terms = _normalize_terms(question)
    answer_terms = _normalize_terms(answer)

    if not question_terms:
        return 0.0

    overlapping_terms = question_terms & answer_terms

    return round(
        len(overlapping_terms) / len(question_terms),
        3,
    )
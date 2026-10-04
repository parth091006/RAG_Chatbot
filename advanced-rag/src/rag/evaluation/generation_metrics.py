from __future__ import annotations


def faithfulness_score(answer: str, evidence: str) -> float:
    if not answer or not evidence:
        return 0.0
    overlap = sum(1 for term in evidence.lower().split() if term in answer.lower().split())
    return round(overlap / max(len(evidence.lower().split()), 1), 3)


def answer_relevance_score(answer: str, question: str) -> float:
    if not answer or not question:
        return 0.0
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    if not question_words:
        return 0.0
    overlap = len(question_words & answer_words)
    return round(overlap / len(question_words), 3)

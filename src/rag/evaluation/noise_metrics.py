"""Exploratory answer-noise diagnostics.

The label-like character-share diagnostic was designed after inspecting the
answers. It is exploratory and must not be retuned before the Phase 6 report.
"""

from __future__ import annotations

import re
from typing import Any

from src.rag.generation.generator import Generator


NO_ANSWER = Generator._fallback_answer()
LABEL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
LABEL_FUNCTION_WORD_RATIO_THRESHOLD = 0.25


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _diagnostic_fragments(text: str) -> list[str]:
    fragments: list[str] = []
    for sentence in _sentences(text):
        fragments.extend(
            part.strip()
            for part in re.split(
                r"(?=\bFigure\s+\d+\s*:)",
                sentence,
                flags=re.IGNORECASE,
            )
            if part.strip()
        )
    return fragments


def function_word_ratio(text: str) -> float:
    words = re.findall(r"\b[\w'-]+\b", text.casefold())
    if not words:
        return 0.0
    return sum(word in LABEL_STOPWORDS for word in words) / len(words)


def label_like_fragment(
    text: str,
    threshold: float = LABEL_FUNCTION_WORD_RATIO_THRESHOLD,
) -> bool:
    normalized = _normalized(text)
    return (
        len(normalized) >= 60
        and not normalized.endswith((".", "!", "?"))
        and function_word_ratio(normalized) < threshold
    )


def document_label_like_stats(
    chunks: list[dict[str, Any]],
    threshold: float = LABEL_FUNCTION_WORD_RATIO_THRESHOLD,
) -> dict[str, Any]:
    ratios = [
        function_word_ratio(fragment)
        for chunk in chunks
        for fragment in _diagnostic_fragments(str(chunk.get("text", "")))
    ]
    flagged = sum(
        label_like_fragment(fragment, threshold)
        for chunk in chunks
        for fragment in _diagnostic_fragments(str(chunk.get("text", "")))
    )
    if not ratios:
        percentiles = {str(percentile): None for percentile in (10, 25, 50, 75, 90, 95)}
    else:
        ordered = sorted(ratios)

        def percentile(value: int) -> float:
            position = (len(ordered) - 1) * value / 100
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (
                position - lower
            )

        percentiles = {
            str(value): percentile(value)
            for value in (10, 25, 50, 75, 90, 95)
        }
    return {
        "fragment_count": len(ratios),
        "flagged_fragment_count": flagged,
        "function_word_ratio_percentiles": percentiles,
        "threshold": threshold,
    }


def _span_char_count(
    spans: object,
    text_length: int,
) -> int | None:
    if spans is None:
        return None
    positions: set[int] = set()
    if isinstance(spans, list):
        for span in spans:
            if not isinstance(span, (tuple, list)) or len(span) != 2:
                continue
            start, end = int(span[0]), int(span[1])
            positions.update(range(max(start, 0), min(end, text_length)))
    return len(positions)


def secondary_char_ratio(chunk: dict[str, Any]) -> float | None:
    value = chunk.get("secondary_char_ratio")
    if value is None:
        return None
    return float(value)


def header_char_ratio(chunk: dict[str, Any]) -> float | None:
    spans = chunk.get("header_spans")
    if spans is None:
        return None
    text = str(chunk.get("text", ""))
    if not text:
        return 0.0
    return (_span_char_count(spans, len(text)) or 0) / len(text)


def header_leak(chunk: dict[str, Any]) -> bool | None:
    ratio = header_char_ratio(chunk)
    if ratio is None:
        return None
    return ratio > 0.0


def header_leak_counts(
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [header_leak(chunk) for chunk in chunks]
    if not values or all(value is None for value in values):
        return {
            "count": None,
            "total": len(values),
            "null": len(values),
        }
    return {
        "count": sum(value is True for value in values),
        "total": len(values),
        "null": sum(value is None for value in values),
    }


def diagram_noise_ratio(
    chunks: list[dict[str, Any]],
    threshold: float = 0.5,
) -> float | None:
    ratios = [
        secondary_char_ratio(chunk)
        for chunk in chunks
        if secondary_char_ratio(chunk) is not None
    ]
    if not ratios:
        return None
    return sum(ratio > threshold for ratio in ratios) / len(ratios)


def context_noise_per_chunk(
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    values = [secondary_char_ratio(chunk) for chunk in chunks]
    present = [value for value in values if value is not None]
    return {
        "values": values,
        "mean": sum(present) / len(present) if present else None,
        "null_excluded": len(values) - len(present),
    }


def _locate_sentence(
    sentence: str,
    contexts: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, int] | None:
    normalized_sentence = _normalized(sentence)
    for context in contexts:
        text = _normalized(str(context.get("text", "")))
        start = text.find(normalized_sentence)
        if start >= 0:
            return context, start, start + len(normalized_sentence)
    return None


def _locate_sentence_fragments(
    sentence: str,
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Locate contiguous answer fragments after generator answer joining."""
    normalized_sentence = _normalized(sentence)
    whole = _locate_sentence(sentence, contexts)
    if whole is not None:
        context, start, end = whole
        return [
            {
                "text": normalized_sentence,
                "context": context,
                "start": start,
                "end": end,
                "located": True,
            }
        ]

    words = normalized_sentence.split()
    fragments: list[dict[str, Any]] = []
    word_index = 0

    while word_index < len(words):
        best: tuple[int, dict[str, Any], int, int] | None = None

        for context in contexts:
            context_text = _normalized(str(context.get("text", "")))
            for end_index in range(
                len(words),
                word_index,
                -1,
            ):
                fragment_text = " ".join(
                    words[word_index:end_index]
                )
                start = context_text.find(fragment_text)
                if start >= 0:
                    candidate = (
                        end_index - word_index,
                        context,
                        start,
                        start + len(fragment_text),
                    )
                    if best is None or candidate[0] > best[0]:
                        best = candidate
                    break

        if best is None:
            fragments.append(
                {
                    "text": words[word_index],
                    "context": None,
                    "start": None,
                    "end": None,
                    "located": False,
                }
            )
            word_index += 1
            continue

        length, context, start, end = best
        fragments.append(
            {
                "text": " ".join(
                    words[word_index:word_index + length]
                ),
                "context": context,
                "start": start,
                "end": end,
                "located": True,
            }
        )
        word_index += length

    return fragments


def answer_fragment_diagnostics(
    answer: str,
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report whole and spliced sentence provenance without changing answers."""
    sentence_reports = []
    total_characters = 0
    unlocated_characters = 0
    secondary_characters = 0
    located_characters = 0

    for sentence in _sentences(answer):
        normalized_sentence = _normalized(sentence)
        fragments = _locate_sentence_fragments(
            sentence,
            contexts,
        )
        located = [
            fragment
            for fragment in fragments
            if fragment["located"]
        ]
        unlocated = [
            fragment
            for fragment in fragments
            if not fragment["located"]
        ]
        sentence_length = len(normalized_sentence)
        sentence_unlocated = sum(
            len(fragment["text"])
            for fragment in unlocated
        )
        total_characters += sentence_length
        unlocated_characters += sentence_unlocated

        for fragment in located:
            fragment_length = int(fragment["end"]) - int(fragment["start"])
            located_characters += fragment_length
            context = fragment["context"]
            for span in context.get("secondary_spans") or []:
                secondary_characters += max(
                    0,
                    min(int(fragment["end"]), int(span[1]))
                    - max(int(fragment["start"]), int(span[0])),
                )

        sentence_reports.append(
            {
                "sentence": sentence,
                "spliced": len(located) > 1,
                "fragments": [
                    {
                        "text": fragment["text"],
                        "chunk_id": (
                            fragment["context"].get("chunk_id")
                            if fragment["context"] is not None
                            else None
                        ),
                        "located": fragment["located"],
                    }
                    for fragment in fragments
                ],
            }
        )

    return {
        "sentences": sentence_reports,
        "spliced_sentence_count": sum(
            report["spliced"]
            for report in sentence_reports
        ),
        "answer_secondary_ratio": (
            secondary_characters / located_characters
            if located_characters
            else None
        ),
        "unlocated_character_share": (
            unlocated_characters / total_characters
            if total_characters
            else 0.0
        ),
    }


def answer_secondary_ratio(
    answer: str,
    contexts: list[dict[str, Any]],
) -> float | None:
    if not contexts or not any(
        secondary_char_ratio(context) is not None
        for context in contexts
    ):
        return None

    return answer_fragment_diagnostics(
        answer,
        contexts,
    )["answer_secondary_ratio"]


def answer_unlocatable_sentence_count(
    answer: str,
    contexts: list[dict[str, Any]],
) -> int:
    return sum(
        _locate_sentence(sentence, contexts) is None
        for sentence in _sentences(answer)
    )


def answer_header_leak(
    answer: str,
    contexts: list[dict[str, Any]],
) -> bool | None:
    if not contexts or not any(
        header_char_ratio(context) is not None
        for context in contexts
    ):
        return None

    for sentence in _sentences(answer):
        found = _locate_sentence(sentence, contexts)
        if found is None:
            continue
        _, start, end = found
        context = found[0]
        for span in context.get("header_spans") or []:
            if start < int(span[1]) and end > int(span[0]):
                return True
    return False


def answer_evidence_support(
    answer: str,
    contexts: list[dict[str, Any]],
) -> float:
    """Measure whole-sentence extractive evidence support, not quality."""
    sentences = [
        sentence
        for sentence in _sentences(answer)
        if _normalized(sentence) != _normalized(NO_ANSWER)
    ]
    if not sentences:
        return 0.0
    context_text = [_normalized(str(context.get("text", ""))) for context in contexts]
    supported = sum(
        _normalized(sentence) in text
        for sentence in sentences
        for text in context_text
    )
    return supported / len(sentences)


def answer_evidence_support_variants(
    answer: str,
    contexts: list[dict[str, Any]],
) -> dict[str, float]:
    """Return whole-sentence and fragment-level evidence support."""
    sentences = [
        sentence
        for sentence in _sentences(answer)
        if _normalized(sentence) != _normalized(NO_ANSWER)
    ]
    if not sentences:
        return {
            "whole_sentence": 0.0,
            "fragment_level": 0.0,
        }

    whole_supported = sum(
        _locate_sentence(sentence, contexts) is not None
        for sentence in sentences
    )
    located_fragments = 0
    supported_fragments = 0
    for sentence in sentences:
        fragments = _locate_sentence_fragments(
            sentence,
            contexts,
        )
        for fragment in fragments:
            located_fragments += 1
            supported_fragments += int(fragment["located"])

    return {
        "whole_sentence": whole_supported / len(sentences),
        "fragment_level": (
            supported_fragments / located_fragments
            if located_fragments
            else 0.0
        ),
    }


def label_like_char_share(
    answer: str,
    contexts: list[dict[str, Any]],
    threshold: float = LABEL_FUNCTION_WORD_RATIO_THRESHOLD,
) -> float:
    """Measure answer characters inside parser-independent label-like fragments."""
    total = 0
    label_like = 0
    for sentence in _sentences(answer):
        for fragment in _locate_sentence_fragments(sentence, contexts):
            for text in _diagnostic_fragments(
                str(fragment["text"])
            ):
                total += len(text)
                if label_like_fragment(text, threshold):
                    label_like += len(text)
    return label_like / total if total else 0.0

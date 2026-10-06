from __future__ import annotations

import re
from typing import TypedDict

from .prompts import qa_prompt


class Source(TypedDict):
    citation_id: str
    document: str
    page: int
    chunk_id: str


class GeneratedAnswer(TypedDict):
    answer: str
    sources: list[Source]


class Generator:
    """Phase-1 deterministic extractive answer generator.

    The implementation intentionally avoids external LLM calls so that
    retrieval quality can be evaluated independently from generation
    quality.

    The public interface is kept compatible with a future LLM-backed
    generator.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    def generate(
        self,
        question: str,
        contexts: list[dict[str, object]],
    ) -> GeneratedAnswer:
        """Generate an answer from the supplied retrieval contexts."""

        prompt = qa_prompt(
            context="\n\n".join(
                str(context["text"])
                for context in contexts
            ),
            question=question,
        )

        answer, selected_contexts = self._generate_answer(
            question=question,
            contexts=contexts,
            prompt=prompt,
        )

        return {
            "answer": answer,
            "sources": self._sources_from_contexts(
                selected_contexts
            ),
        }

    def _generate_answer(
        self,
        question: str,
        contexts: list[dict[str, object]],
        prompt: str,
    ) -> tuple[str, list[dict[str, object]]]:
        """Select useful evidence sentences deterministically.

        The prompt is intentionally unused during Phase 1. It remains part
        of the internal interface so a future LLM implementation can use
        the same Generator API.

        Sentence selection uses:
        - question-term overlap
        - question-term coverage
        - reranked context priority
        - duplicate suppression
        """

        del prompt

        question_terms = self._extract_terms(question)

        if not question_terms:
            return (
                self._fallback_answer(),
                [],
            )

        candidates: list[
            tuple[float, int, int, str]
        ] = []

        for context_index, context in enumerate(contexts):
            text = str(context["text"])

            sentences = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

            context_priority = 1.0 / (context_index + 1)

            for sentence_index, sentence in enumerate(sentences):
                sentence = sentence.strip()

                if not sentence:
                    continue

                sentence_terms = self._extract_terms(sentence)

                if not sentence_terms:
                    continue

                overlap_terms = question_terms & sentence_terms

                if not overlap_terms:
                    continue

                overlap = len(overlap_terms)

                coverage = overlap / len(question_terms)

                score = (
                    overlap
                    + coverage
                    + context_priority
                )

                candidates.append(
                    (
                        score,
                        context_index,
                        sentence_index,
                        sentence,
                    )
                )

        candidates.sort(
            key=lambda item: (
                item[0],
                -item[1],
                -item[2],
            ),
            reverse=True,
        )

        selected_sentences: list[str] = []
        selected_contexts: list[dict[str, object]] = []
        seen_sentences: set[str] = set()
        seen_contexts: set[int] = set()

        for _, context_index, _, sentence in candidates:
            normalized = self._normalize_sentence(sentence)

            if normalized in seen_sentences:
                continue

            if self._is_near_duplicate(
                normalized,
                seen_sentences,
            ):
                continue

            seen_sentences.add(normalized)

            selected_sentences.append(sentence)

            if context_index not in seen_contexts:
                selected_contexts.append(
                    contexts[context_index]
                )
                seen_contexts.add(context_index)

            if len(selected_sentences) == 3:
                break

        if not selected_sentences:
            return (
                self._fallback_answer(),
                [],
            )

        return (
            " ".join(selected_sentences),
            selected_contexts,
        )

    @staticmethod
    def _extract_terms(text: str) -> set[str]:
        """Extract normalized terms used by the lexical baseline."""

        return {
            term.lower()
            for term in re.findall(
                r"\b[\w'-]+\b",
                text,
            )
            if len(term) > 2
        }

    @staticmethod
    def _normalize_sentence(sentence: str) -> str:
        """Normalize a sentence for duplicate detection."""

        return re.sub(
            r"\s+",
            " ",
            sentence.lower(),
        ).strip()

    @staticmethod
    def _is_near_duplicate(
        sentence: str,
        seen_sentences: set[str],
    ) -> bool:
        """Detect highly overlapping evidence sentences."""

        sentence_terms = Generator._extract_terms(sentence)

        if not sentence_terms:
            return True

        for seen in seen_sentences:
            seen_terms = Generator._extract_terms(seen)

            if not seen_terms:
                continue

            intersection = len(
                sentence_terms & seen_terms
            )

            smaller_set = min(
                len(sentence_terms),
                len(seen_terms),
            )

            if smaller_set == 0:
                continue

            overlap_ratio = intersection / smaller_set

            if overlap_ratio >= 0.8:
                return True

        return False

    @staticmethod
    def _fallback_answer() -> str:
        return (
            "I could not find sufficient evidence in the "
            "provided document to answer this question."
        )

    @staticmethod
    def _sources_from_contexts(
        contexts: list[dict[str, object]],
    ) -> list[Source]:
        sources: list[Source] = []

        for context in contexts:
            document = str(context["document_name"])
            page = int(context["page_number"])
            chunk_id = str(context["chunk_id"])

            sources.append(
                {
                    "citation_id": f"[{chunk_id}]",
                    "document": document,
                    "page": page,
                    "chunk_id": chunk_id,
                }
            )

        return sources
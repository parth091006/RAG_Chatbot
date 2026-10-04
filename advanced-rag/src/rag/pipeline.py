from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .chunking.chunker import chunk_document
from .embeddings.embedder import Embedder
from .generation.generator import Generator
from .ingestion.parser import extract_text_from_pdf


class PhaseOneRAG:
    """Minimal end-to-end phase-1 pipeline: PDF -> text -> chunks -> embeddings -> retrieval -> answer."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.text = ""
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []
        self.embedder = Embedder()
        self.generator = Generator()

    def pdf_to_text(self) -> str:
        self.text = extract_text_from_pdf(self.pdf_path)
        return self.text

    def pdf_to_chunks(self, chunk_size: int = 250, overlap: int = 50) -> list[str]:
        if not self.text:
            self.pdf_to_text()
        self.chunks = chunk_document(self.text, chunk_size=chunk_size, overlap=overlap)
        return self.chunks

    def chunks_to_embeddings(self, chunks: list[str] | None = None) -> list[list[float]]:
        chunk_list = chunks if chunks is not None else self.chunks
        if not chunk_list:
            chunk_list = self.pdf_to_chunks()
        self.embeddings = self.embedder.embed_many(chunk_list)
        return self.embeddings

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"\b[\w'-]+\b", text) if len(token) > 2]

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def question_to_top_k_chunks(self, question: str, top_k: int = 5) -> list[str]:
        if not self.chunks:
            self.pdf_to_chunks()
        if not self.embeddings:
            self.chunks_to_embeddings(self.chunks)

        query_vector = self.embedder.embed(question)
        scored = []
        for idx, chunk in enumerate(self.chunks):
            score = self._cosine_similarity(query_vector, self.embeddings[idx])
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [chunk for _, chunk in scored[:top_k]]
        return top_chunks

    def answer_question(self, question: str, top_k: int = 5) -> str:
        retrieved = self.question_to_top_k_chunks(question, top_k=top_k)
        if not retrieved:
            return "I could not find sufficient evidence in the provided document to answer this question."

        context = "\n\n".join(f"Context {idx + 1}: {chunk}" for idx, chunk in enumerate(retrieved))
        return self.generator.generate_from_context(question, context)

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking.chunker import Chunk, chunk_pages, section_aware_chunks
from .embeddings.embedder import Embedder
from .generation.generator import Generator
from .ingestion.parser import extract_text_from_pdf
from .retrieval.dense import DenseRetriever


class PhaseOneRAG:
    """Phase-1 RAG pipeline for ingestion, retrieval, and extractive generation.

    The pipeline intentionally keeps the architecture simple so retrieval
    experiments can be evaluated independently.

    Supported embedding methods:
        - bow
        - tfidf
        - semantic

    Retrieval in the main Phase-1 pipeline uses the DenseRetriever interface
    over whichever representation the Embedder produces.

    BM25 and hybrid retrieval are implemented separately and can be evaluated
    as experiments without changing this core pipeline.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        embedding_method: str = "bow",
        semantic_model_name: str | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.embedding_method = embedding_method

        self.embedder = Embedder(
            model_name=embedding_method,
            semantic_model_name=semantic_model_name,
        )
        self.generator = Generator()

        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []
        self._retriever: DenseRetriever | None = None

    def pdf_to_text(self) -> str:
        """Extract text from the configured PDF."""
        return extract_text_from_pdf(self.pdf_path)

    def pdf_to_chunks(
        self,
        strategy: str = "fixed",
        chunk_size: int = 250,
        overlap: int = 50,
    ) -> list[Chunk]:
        """Convert the configured PDF into retrieval chunks.

        Args:
            strategy:
                ``"fixed"`` for the baseline fixed-size chunking strategy or
                ``"section"`` for section-aware chunking.
            chunk_size:
                Maximum number of words per chunk.
            overlap:
                Number of overlapping words between consecutive chunks.

        Returns:
            List of document chunks.
        """
        pages = self._load_pdf_pages()

        if strategy == "fixed":
            chunks = chunk_pages(
                pages,
                document_id=self._document_id(),
                document_name=self.pdf_path.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        elif strategy == "section":
            chunks = section_aware_chunks(
                pages,
                document_id=self._document_id(),
                document_name=self.pdf_path.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        else:
            raise ValueError(
                f"Unsupported chunking strategy: {strategy!r}. "
                "Use 'fixed' or 'section'."
            )

        self._chunks = chunks
        self._embeddings = []
        self._retriever = None

        return chunks

    def chunks_to_embeddings(
        self,
        chunks: list[Chunk] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings/representations for document chunks."""
        if chunks is None:
            chunks = self._chunks

        if not chunks:
            raise ValueError(
                "No chunks available. Call pdf_to_chunks() first."
            )

        texts = [str(chunk["text"]) for chunk in chunks]
        embeddings = self.embedder.embed_many(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        self._chunks = chunks
        self._embeddings = embeddings
        self._retriever = DenseRetriever(embeddings)

        return embeddings

    def question_to_scored_chunks(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k chunks together with their retrieval scores.

        Returns:
            A list of dictionaries containing:
                - ``chunk``: the original Chunk
                - ``score``: cosine similarity score
        """
        if top_k <= 0:
            return []

        self._ensure_index()

        query_embedding = self.embedder.embed(question)

        assert self._retriever is not None

        ranked_results = self._retriever.search(
            query_embedding,
            top_k=top_k,
        )

        return [
            {
                "chunk": self._chunks[index],
                "score": float(score),
            }
            for index, score in ranked_results
        ]

    def question_to_top_k_chunks(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[Chunk]:
        """Retrieve the top-k document chunks for a question."""
        scored_chunks = self.question_to_scored_chunks(
            question,
            top_k=top_k,
        )

        return [
            result["chunk"]
            for result in scored_chunks
        ]

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve evidence and generate an extractive answer."""
        contexts = self.question_to_top_k_chunks(
            question,
            top_k=top_k,
        )

        return self.generator.generate(
            question,
            contexts,
        )

    def _ensure_index(self) -> None:
        """Ensure chunks, embeddings, and the retriever are initialized."""
        if not self._chunks:
            self.pdf_to_chunks()

        if not self._embeddings:
            self.chunks_to_embeddings(self._chunks)

        if self._retriever is None:
            self._retriever = DenseRetriever(self._embeddings)

    def _load_pdf_pages(self) -> list[str]:
        """Load PDF text while preserving page boundaries."""
        from pypdf import PdfReader

        reader = PdfReader(str(self.pdf_path))

        return [
            page.extract_text() or ""
            for page in reader.pages
        ]

    def _document_id(self) -> str:
        """Create the document ID used by the chunking layer."""
        return f"doc_{self.pdf_path.stem}"
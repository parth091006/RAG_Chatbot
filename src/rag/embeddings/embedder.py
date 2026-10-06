from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


class Embedder:
    """Configurable embedding/representation model for retrieval experiments.

    Supported methods:
    - bow: deterministic normalized bag-of-words
    - tfidf: TF-IDF weighted lexical representation
    - semantic: sentence-transformers semantic embeddings

    The interface is intentionally kept the same across all methods so that
    retrieval and evaluation can compare the representations without changing
    the rest of the RAG pipeline.
    """

    SUPPORTED_METHODS = {"bow", "tfidf", "semantic"}

    # Small, general-purpose sentence embedding model.
    # The model is downloaded automatically by sentence-transformers the first
    # time semantic embeddings are used.
    DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = "bow",
        semantic_model_name: str | None = None,
    ):
        if model_name not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported embedding method: {model_name!r}. "
                f"Choose one of: {sorted(self.SUPPORTED_METHODS)}"
            )

        self.model_name = model_name
        self.vocabulary: list[str] = []
        self.idf: dict[str, float] = {}
        self.document_count = 0

        self.semantic_model_name = (
            semantic_model_name or self.DEFAULT_SEMANTIC_MODEL
        )
        self._semantic_model = None

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in re.findall(r"\b[\w'-]+\b", text)
            if len(token) > 2
        ]

    def _load_semantic_model(self):
        """Load the sentence-transformers model only when semantic retrieval is used."""
        if self._semantic_model is not None:
            return self._semantic_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Semantic embeddings require the 'sentence-transformers' "
                "package. Install it with: pip install sentence-transformers"
            ) from exc

        self._semantic_model = SentenceTransformer(
            self.semantic_model_name
        )
        return self._semantic_model

    def fit(self, texts: Sequence[str]) -> list[str]:
        """Fit the representation on the document collection.

        Semantic embeddings do not require fitting because the pretrained
        sentence-transformers model already contains its representation space.
        """
        if self.model_name == "semantic":
            self.document_count = len(texts)
            return []

        tokenized_documents = [self._tokenize(text) for text in texts]

        vocabulary = sorted(
            {
                token
                for tokens in tokenized_documents
                for token in tokens
            }
        )

        self.vocabulary = vocabulary
        self.document_count = len(tokenized_documents)

        if self.model_name == "tfidf":
            document_frequency = Counter()

            for tokens in tokenized_documents:
                unique_tokens = set(tokens)
                document_frequency.update(unique_tokens)

            # Smoothed IDF:
            #
            # idf(t) = log((N + 1) / (df(t) + 1)) + 1
            #
            # Smoothing prevents division/log problems and keeps
            # every IDF value positive.
            self.idf = {
                token: math.log(
                    (self.document_count + 1)
                    / (document_frequency[token] + 1)
                )
                + 1.0
                for token in self.vocabulary
            }
        else:
            self.idf = {}

        return self.vocabulary

    def _term_frequency(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1

        return {
            token: count / total
            for token, count in counts.items()
        }

    def _embed_lexical(self, text: str) -> list[float]:
        """Create a BoW or TF-IDF vector."""
        if not self.vocabulary:
            self.fit([text])

        tokens = self._tokenize(text)
        term_frequency = self._term_frequency(tokens)

        if self.model_name == "bow":
            return [
                term_frequency.get(token, 0.0)
                for token in self.vocabulary
            ]

        return [
            term_frequency.get(token, 0.0)
            * self.idf.get(token, 1.0)
            for token in self.vocabulary
        ]

    def _embed_semantic(self, text: str) -> list[float]:
        """Create a semantic embedding using a pretrained model."""
        model = self._load_semantic_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed(self, text: str) -> list[float]:
        """Convert text into the configured vector representation."""
        if self.model_name == "semantic":
            return self._embed_semantic(text)

        return self._embed_lexical(text)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Fit the representation on documents and embed every document."""
        if not texts:
            return []

        if self.model_name == "semantic":
            model = self._load_semantic_model()

            embeddings = model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return [embedding.tolist() for embedding in embeddings]

        self.fit(texts)

        return [self.embed(text) for text in texts]
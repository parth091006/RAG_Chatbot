from __future__ import annotations

import re
from collections import Counter
from typing import Sequence


class Embedder:
    """Deterministic bag-of-words embeddings for the phase-1 pipeline."""

    def __init__(self, model_name: str = "deterministic-bow"):
        self.model_name = model_name
        self.vocabulary: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"\b[\w'-]+\b", text) if len(token) > 2]

    def fit(self, texts: Sequence[str]) -> list[str]:
        vocabulary = sorted({token for text in texts for token in self._tokenize(text)})
        self.vocabulary = vocabulary
        return vocabulary

    def embed(self, text: str) -> list[float]:
        if not self.vocabulary:
            self.fit([text])

        counts = Counter(self._tokenize(text))
        total = sum(counts.values()) or 1
        return [counts.get(token, 0.0) / total for token in self.vocabulary]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self.fit(texts)
        return [self.embed(text) for text in texts]

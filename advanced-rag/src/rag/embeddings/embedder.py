from __future__ import annotations

from typing import Sequence


class Embedder:
    """A small wrapper around an embedding model interface."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        """Return a deterministic placeholder embedding for text."""
        return [float(len(text)), float(text.count(" ") + 1), float(len(set(text.lower())))]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

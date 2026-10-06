from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    """Second-stage cross-encoder reranker.

    The reranker scores each (query, document) pair jointly using a
    cross-encoder model. This is different from embedding-based retrieval:
    the query and candidate document are evaluated together, allowing
    the model to capture their interaction more directly.

    Intended workflow:

        Retriever
            ↓
        Top-N candidate chunks
            ↓
        Cross-Encoder Reranker
            ↓
        Top-K chunks
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        self.model_name = model_name
        self.top_k = top_k

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[int, str]],
    ) -> list[tuple[int, float]]:
        """Rerank candidate documents for a query.

        Args:
            query:
                The user's search question.

            candidates:
                Candidate documents represented as:
                    (document_index, document_text)

        Returns:
            A list of:
                (document_index, reranker_score)

            ordered from highest to lowest reranker score.
        """

        if not candidates:
            return []

        pairs = [
            [query, document_text]
            for _, document_text in candidates
        ]

        scores = self.model.predict(pairs)

        ranked = [
            (index, float(score))
            for (index, _), score in zip(candidates, scores)
        ]

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked[: self.top_k]
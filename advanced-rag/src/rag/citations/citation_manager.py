from __future__ import annotations


class CitationManager:
    def __init__(self):
        self._map: dict[str, dict[str, str]] = {}

    def register(self, citation_id: str, document: str, page: str, chunk_id: str) -> None:
        self._map[citation_id] = {
            "document": document,
            "page": page,
            "chunk_id": chunk_id,
        }

    def resolve(self, citation_id: str) -> dict[str, str]:
        return self._map.get(citation_id, {"document": "unknown", "page": "unknown", "chunk_id": "unknown"})

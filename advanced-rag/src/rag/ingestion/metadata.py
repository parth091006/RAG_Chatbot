from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentMetadata:
    document_id: str
    filename: str
    file_path: str
    pages: int = 0
    language: str | None = None
    authors: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, file_path: str | Path, document_id: str | None = None) -> "DocumentMetadata":
        path = Path(file_path)
        filename = path.name
        _document_id = document_id or f"doc_{filename}"
        return cls(
            document_id=_document_id,
            filename=filename,
            file_path=str(path),
            pages=0,
            authors=[],
        )

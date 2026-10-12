from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hash of a file reading in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class DocumentMetadata:
    """Metadata container for ingested documents."""

    document_id: str
    filename: str
    file_path: str
    sha256_hash: str = ""
    file_size_bytes: int = 0
    pages: int = 0
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    subject: str | None = None
    keywords: list[str] = field(default_factory=list)
    creation_date: str | None = None
    language: str | None = None
    custom: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        document_id: str | None = None,
        custom: dict[str, Any] | None = None,
    ) -> DocumentMetadata:
        """Create DocumentMetadata from a PDF or document file."""
        path = Path(file_path)
        filename = path.name
        _document_id = document_id or f"doc_{path.stem}"

        pages = 0
        sha256_hash = ""
        file_size = 0
        title: str | None = None
        authors: list[str] = []
        subject: str | None = None
        keywords: list[str] = []
        creation_date: str | None = None

        if path.exists():
            file_size = path.stat().st_size
            sha256_hash = compute_sha256(path)

            if path.suffix.lower() == ".pdf":
                try:
                    import pymupdf

                    doc = pymupdf.open(path)
                    pages = len(doc)
                    raw_meta = doc.metadata or {}
                    doc.close()

                    raw_title = raw_meta.get("title")
                    if raw_title and raw_title.strip():
                        title = raw_title.strip()

                    raw_author = raw_meta.get("author")
                    if raw_author and raw_author.strip():
                        # Authors may be separated by commas, semicolons, or 'and'
                        raw_authors = [
                            a.strip()
                            for a in raw_author.replace(";", ",").split(",")
                            if a.strip()
                        ]
                        authors = raw_authors

                    raw_subject = raw_meta.get("subject")
                    if raw_subject and raw_subject.strip():
                        subject = raw_subject.strip()

                    raw_keywords = raw_meta.get("keywords")
                    if raw_keywords and raw_keywords.strip():
                        keywords = [
                            k.strip()
                            for k in raw_keywords.replace(";", ",").split(",")
                            if k.strip()
                        ]

                    raw_creation = raw_meta.get("creationDate")
                    if raw_creation and raw_creation.strip():
                        creation_date = raw_creation.strip()

                except Exception:
                    pass

        return cls(
            document_id=_document_id,
            filename=filename,
            file_path=str(path),
            sha256_hash=sha256_hash,
            file_size_bytes=file_size,
            pages=pages,
            title=title,
            authors=authors,
            subject=subject,
            keywords=keywords,
            creation_date=creation_date,
            custom=custom or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        """Construct DocumentMetadata from a dictionary."""
        known_fields = {
            "document_id",
            "filename",
            "file_path",
            "sha256_hash",
            "file_size_bytes",
            "pages",
            "title",
            "authors",
            "subject",
            "keywords",
            "creation_date",
            "language",
            "custom",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

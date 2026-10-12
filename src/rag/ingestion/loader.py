from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metadata import DocumentMetadata, compute_sha256


def load_pdf_paths(directory: str | Path) -> list[Path]:
    """Return a sorted list of PDF files in a directory."""
    folder = Path(directory)
    if not folder.exists():w
        return []
    return sorted(folder.glob("*.pdf"))


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute the SHA-256 hash of a file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return compute_sha256(path)


@dataclass
class IngestionReport:
    """Report detailing the outcome of a document ingestion run."""

    documents: list[DocumentMetadata] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def total_scanned(self) -> int:
        return len(self.documents) + len(self.duplicates) + len(self.errors)

    @property
    def ingested_count(self) -> int:
        return len(self.documents)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicates)


class DocumentLoader:
    """Manages document ingestion, metadata extraction, and duplicate detection."""

    def __init__(self) -> None:
        self._by_hash: dict[str, DocumentMetadata] = {}
        self._by_id: dict[str, DocumentMetadata] = {}

    def is_duplicate_hash(self, sha256_hash: str) -> bool:
        """Check if a SHA-256 hash has already been ingested."""
        return sha256_hash in self._by_hash

    def is_duplicate_file(self, file_path: str | Path) -> bool:
        """Check if a file content matches an already ingested file."""
        path = Path(file_path)
        if not path.exists():
            return False
        return self.is_duplicate_hash(compute_sha256(path))

    def load_file(
        self,
        file_path: str | Path,
        deduplicate: bool = True,
        document_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> tuple[DocumentMetadata, bool]:
        """Load and extract metadata for a single file.

        Returns:
            A tuple of (metadata, is_duplicate).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        meta = DocumentMetadata.from_file(
            path,
            document_id=document_id,
            custom=custom_metadata,
        )

        is_dup = meta.sha256_hash in self._by_hash

        if is_dup and deduplicate:
            existing = self._by_hash[meta.sha256_hash]
            return existing, True

        self._by_hash[meta.sha256_hash] = meta
        self._by_id[meta.document_id] = meta
        return meta, False

    def load_directory(
        self,
        directory: str | Path,
        deduplicate: bool = True,
        pattern: str = "*.pdf",
    ) -> IngestionReport:
        """Scan a directory, extract metadata, and identify duplicate files."""
        folder = Path(directory)
        report = IngestionReport()

        if not folder.exists():
            return report

        pdf_files = sorted(folder.glob(pattern))

        for file_path in pdf_files:
            try:
                meta = DocumentMetadata.from_file(file_path)

                if meta.sha256_hash in self._by_hash and deduplicate:
                    existing = self._by_hash[meta.sha256_hash]
                    report.duplicates.append(
                        {
                            "duplicate_file": str(file_path),
                            "original_file": existing.file_path,
                            "original_document_id": existing.document_id,
                            "sha256_hash": meta.sha256_hash,
                        }
                    )
                else:
                    self._by_hash[meta.sha256_hash] = meta
                    self._by_id[meta.document_id] = meta
                    report.documents.append(meta)

            except Exception as exc:
                report.errors.append(
                    {
                        "file": str(file_path),
                        "error": str(exc),
                    }
                )

        return report

    def get_document(self, document_id: str) -> DocumentMetadata | None:
        """Retrieve document metadata by document ID."""
        return self._by_id.get(document_id)

    def get_by_hash(self, sha256_hash: str) -> DocumentMetadata | None:
        """Retrieve document metadata by SHA-256 hash."""
        return self._by_hash.get(sha256_hash)

    def list_documents(self) -> list[DocumentMetadata]:
        """Return all tracked documents."""
        return list(self._by_id.values())

    def reset(self) -> None:
        """Clear the registry."""
        self._by_hash.clear()
        self._by_id.clear()

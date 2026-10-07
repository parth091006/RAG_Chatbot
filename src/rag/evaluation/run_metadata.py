from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.rag.chunking.chunker import Chunk


EXTRACTION_CHOICES = (
    "legacy-pypdf",
    "current",
    "primary-only",
)
OPTIONAL_CHUNK_KEYS = (
    "chunk_text_hash",
    "secondary_char_ratio",
    "secondary_spans",
    "header_spans",
)


def resolve_extraction(
    extraction: str | None,
    header_fix: bool,
    default: str,
) -> tuple[str, bool]:
    """Resolve CLI extraction options and whether metadata is enabled."""
    if extraction is not None and extraction not in EXTRACTION_CHOICES:
        raise ValueError(
            f"Unsupported extraction: {extraction!r}. "
            f"Choose one of: {list(EXTRACTION_CHOICES)}"
        )

    if header_fix and extraction not in {"current", "primary-only"}:
        raise ValueError(
            "--header-fix requires --extraction current or primary-only."
        )

    return extraction or default, extraction is not None or header_fix


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip() or None


def build_run_metadata(
    *,
    extraction: str,
    header_fix: bool,
    embedding_method: str,
    chunk_size: int,
    overlap: int,
    pipeline: Any,
    pdf_path: Path,
    question_path: Path | None = None,
    candidate_k: int | None = None,
    top_k: int | None = None,
    reranker_model: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "extraction": extraction,
        "header_fix": header_fix,
        "embedding_method": embedding_method,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "reranker_model": reranker_model,
        "chunk_count": pipeline.chunk_count,
        "total_page_characters": pipeline.total_page_characters,
        "total_chunk_characters": pipeline.total_chunk_characters,
        "pdf_path": str(pdf_path),
        "pdf_sha256": sha256_file(pdf_path),
        "question_file_path": str(question_path) if question_path else None,
        "question_file_sha256": (
            sha256_file(question_path)
            if question_path is not None
            else None
        ),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }
    return metadata


def serialize_chunk(
    chunk: Chunk,
    *,
    include_optional: bool = False,
) -> dict[str, Any]:
    """Serialize a chunk without optional experiment-only fields by default."""
    serialized = dict(chunk)

    if include_optional:
        serialized.pop("secondary_spans", None)
        serialized.pop("header_spans", None)
        serialized.setdefault("secondary_char_ratio", None)
        return serialized

    for key in OPTIONAL_CHUNK_KEYS:
        serialized.pop(key, None)

    return serialized


def safe_write_json(
    output_path: Path,
    payload: Any,
    *,
    metadata: dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    """Write JSON, guarding only explicitly configured experiment outputs."""
    if metadata is not None and output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output: {output_path}. "
            "Use --force to overwrite it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata is not None and isinstance(payload, dict):
        payload = dict(payload)
        payload["run_metadata"] = metadata

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if metadata is not None and isinstance(payload, list):
        metadata_path = output_path.with_name(
            output_path.name + ".meta.json"
        )
        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

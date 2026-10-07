import json
from pathlib import Path

import pytest

from src.rag.evaluation.run_metadata import (
    resolve_extraction,
    safe_write_json,
    serialize_chunk,
)
from src.rag.pipeline import PhaseOneRAG


PDF_PATH = Path("data/raw/self-attention-transformers-2023.pdf")


def _saved_chunk(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload["results"][0]["results"][0]["chunk"]
    return payload[0]["results"][0]["chunk"]


@pytest.mark.parametrize(
    "result_path",
    [
        Path("data/evaluation/retrieval_results.json"),
        Path("data/evaluation/reranker_results.json"),
        Path("data/evaluation/hybrid_retrieval_results.json"),
    ],
)
def test_no_flag_chunk_keys_match_saved_results(
    result_path: Path,
) -> None:
    pipeline = PhaseOneRAG(PDF_PATH)
    chunks = pipeline.pdf_to_chunks()
    pipeline.chunks_to_embeddings(chunks)

    saved_keys = set(_saved_chunk(result_path))
    no_flag_keys = set(serialize_chunk(chunks[0]))

    assert no_flag_keys == saved_keys


def test_optional_chunk_serialization_excludes_secondary_spans() -> None:
    chunk = {
        "chunk_id": "doc_chunk_001",
        "document_id": "doc",
        "document_name": "paper.pdf",
        "page_number": 1,
        "text": "text",
        "chunk_text_hash": "123456789abc",
        "secondary_char_ratio": 0.25,
        "secondary_spans": [(0, 4)],
    }

    serialized = serialize_chunk(chunk, include_optional=True)

    assert serialized["chunk_text_hash"] == "123456789abc"
    assert serialized["secondary_char_ratio"] == 0.25
    assert "secondary_spans" not in serialized


def test_extraction_resolution_and_header_fix_validation() -> None:
    assert resolve_extraction(
        None,
        False,
        "legacy-pypdf",
    ) == ("legacy-pypdf", False)
    assert resolve_extraction(
        "current",
        False,
        "legacy-pypdf",
    ) == ("current", True)
    assert resolve_extraction(
        "primary-only",
        True,
        "legacy-pypdf",
    ) == ("primary-only", True)

    with pytest.raises(ValueError, match="current or primary-only"):
        resolve_extraction(
            None,
            True,
            "legacy-pypdf",
        )


def test_safe_write_json_guards_and_writes_metadata_sidecar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results.json"
    metadata = {"extraction": "current", "header_fix": False}

    safe_write_json(
        output,
        [{"result": 1}],
        metadata=metadata,
    )

    sidecar = Path(str(output) + ".meta.json")
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"result": 1}
    ]
    assert json.loads(sidecar.read_text(encoding="utf-8")) == metadata

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        safe_write_json(
            output,
            [{"result": 2}],
            metadata=metadata,
        )

    safe_write_json(
        output,
        [{"result": 2}],
        metadata=metadata,
        force=True,
    )

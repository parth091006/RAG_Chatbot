from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_parser_module_requires_pdf_path() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.rag.ingestion.parser"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode != 0
    assert "pdf_path" in result.stderr

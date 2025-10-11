from __future__ import annotations

from pathlib import Path

import pytest

from mosaicx.extractor import ExtractionError, extract_text_from_document, load_schema_model


def test_extract_text_from_plain_text(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("Patient: Demo", encoding="utf-8")

    result = extract_text_from_document(note)
    assert "Patient" in result


def test_extract_text_rejects_unknown_extension(tmp_path: Path) -> None:
    note = tmp_path / "note.xyz"
    note.write_text("content", encoding="utf-8")
    with pytest.raises(ExtractionError):
        extract_text_from_document(note)


def test_load_schema_model_missing(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError):
        load_schema_model(str(tmp_path / "missing.py"))

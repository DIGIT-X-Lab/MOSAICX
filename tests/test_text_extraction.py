from __future__ import annotations

from pathlib import Path

import pytest

from mosaicx.text_extraction import LayeredTextResult, TextExtractionError, extract_text_with_fallback


@pytest.fixture
def plain_text_file(tmp_path: Path) -> Path:
    txt = tmp_path / "note.txt"
    txt.write_text("Patient: Demo", encoding="utf-8")
    return txt


def test_plain_text_bypasses_docling(plain_text_file: Path) -> None:
    result = extract_text_with_fallback(plain_text_file)
    assert isinstance(result, LayeredTextResult)
    assert result.mode == "plain"
    assert result.markdown == "Patient: Demo"


def test_unknown_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "note.xyz"
    path.write_text("Unsupported", encoding="utf-8")
    with pytest.raises(TextExtractionError):
        extract_text_with_fallback(path)

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel

from mosaicx.extractor import (
    ExtractionError,
    extract_structured_data,
    extract_text_from_document,
    load_schema_model,
)
from mosaicx.prompting import build_example_template


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


def test_prompt_override_requires_dspy_strategy() -> None:
    class TinySchema(BaseModel):
        value: int  # type: ignore[annotation-unchecked]

    with pytest.raises(ExtractionError):
        extract_structured_data(
            text_content="value: 1",
            schema_class=TinySchema,
            prompt_path=Path("dummy_prompt.txt"),
        )

    with pytest.raises(ExtractionError):
        extract_structured_data(
            text_content="value: 1",
            schema_class=TinySchema,
            prompt_preference="base",
        )

    with pytest.raises(ExtractionError):
        extract_structured_data(
            text_content="value: 1",
            schema_class=TinySchema,
            dspy_examples_path=Path("train.json"),
        )


def test_build_example_template_structure() -> None:
    class MiniSchema(BaseModel):
        flag: Optional[int] = None

    template = build_example_template(MiniSchema, text_placeholder="<text>")
    assert template["text"] == "<text>"
    assert "json_output" in template
    assert template["json_output"].get("flag") is None

"""Unit tests for the public MOSAICX Python API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from mosaicx import generate_schema, extract_pdf, summarize_reports
from mosaicx.summarizer import PatientSummary, PatientHeader


class DummyModel(BaseModel):
    name: str


@pytest.fixture()
def dummy_schema_file(tmp_path: Path) -> Path:
    schema_code = """from pydantic import BaseModel\n\nclass Dummy(BaseModel):\n    name: str\n"""
    path = tmp_path / "dummy_schema.py"
    path.write_text(schema_code)
    return path


def test_generate_schema_accepts_sequence_and_normalises_regex(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_synth(description: str, **kwargs: Any) -> str:
        recorded["description"] = description
        return "from pydantic import BaseModel, Field\n\nclass Foo(BaseModel):\n    name: str = Field(..., regex='^foo$')\n"

    monkeypatch.setattr("mosaicx.api.schema.synthesize_pydantic_model", fake_synth)

    schema = generate_schema(["Part A", "Part B"], class_name="Foo")

    assert recorded["description"] == "Part A\nPart B"
    assert "pattern='^foo$'" in schema.code

    target = schema.write("schemas/foo.py")
    assert target.name == "foo.py"


@pytest.fixture()
def patch_extract_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_schema_class = DummyModel

    def fake_load_schema_model(_: str) -> type[BaseModel]:
        return dummy_schema_class

    def fake_extract_text(_: Path) -> str:
        return "Patient: Alice"

    def fake_extract_structured(text: str, schema_cls: type[BaseModel], **_: Any) -> BaseModel:
        assert schema_cls is dummy_schema_class
        return schema_cls(name=text.split(":")[-1].strip())

    monkeypatch.setattr("mosaicx.api.extraction.load_schema_model", fake_load_schema_model)
    monkeypatch.setattr("mosaicx.api.extraction.extract_text_from_pdf", fake_extract_text)
    monkeypatch.setattr("mosaicx.api.extraction.extract_structured_data", fake_extract_structured)


def test_extract_pdf_accepts_string_paths(tmp_path: Path, patch_extract_dependencies: None, dummy_schema_file: Path) -> None:
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_text("dummy pdf")

    result = extract_pdf(str(pdf_path), str(dummy_schema_file))

    assert result.schema_path == dummy_schema_file
    assert result.pdf_path == pdf_path
    assert result.to_dict()["name"] == "Alice"

    out_path = tmp_path / "out.json"
    result.write_json(out_path)
    saved = json.loads(out_path.read_text())
    assert saved["name"] == "Alice"


def test_summarize_reports_accepts_string_and_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text_file = tmp_path / "note.txt"
    text_file.write_text("Example report")

    captured_paths: list[Path] = []

    def fake_load_reports(paths: list[Path]) -> list[str]:
        captured_paths.extend(paths)
        return ["doc"]

    def fake_summarize_with_llm(docs: list[str], **_: Any) -> PatientSummary:
        assert docs == ["doc"]
        return PatientSummary(
            patient=PatientHeader(patient_id="demo"),
            timeline=[],
            overall="ok",
        )

    monkeypatch.setattr("mosaicx.api.summary.load_reports", fake_load_reports)
    monkeypatch.setattr("mosaicx.api.summary.summarize_with_llm", fake_summarize_with_llm)

    summarize_reports([str(text_file.parent)], patient_id="demo")

    assert text_file in captured_paths

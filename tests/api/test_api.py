from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from mosaicx.api import extract_document, generate_schema, summarize_reports


class DummyModel(BaseModel):
    name: str


@pytest.fixture
def dummy_schema_file(tmp_path: Path) -> Path:
    schema_code = """
from pydantic import BaseModel

class Dummy(BaseModel):
    name: str
"""
    path = tmp_path / "dummy_schema.py"
    path.write_text(schema_code, encoding="utf-8")
    return path


def test_generate_schema_combines_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    def fake_synth(*, description: str, **kwargs: Any) -> str:
        recorded["desc"] = description
        return "from pydantic import BaseModel\n\nclass Foo(BaseModel):\n    name: str\n"

    monkeypatch.setattr("mosaicx.api.schema.synthesize_pydantic_model", fake_synth)

    schema = generate_schema(["Part A", "Part B"], class_name="Foo")
    assert recorded["desc"] == "Part A\nPart B"
    target = schema.write("schemas/foo.py")
    assert target.name == "foo.py"


@pytest.fixture
def patch_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_schema_model(_: str) -> type[BaseModel]:
        return DummyModel

    from mosaicx.text_extraction import LayeredTextResult

    def fake_extract_text(_: Path, **__: Any) -> LayeredTextResult:
        return LayeredTextResult(
            markdown="Patient: Alice",
            mode="native",
            page_analysis=[],
            attempts=["native"],
            vlm_pages=[],
        )

    def fake_extract_structured(text: str, schema_cls: type[BaseModel], **__: Any) -> BaseModel:
        assert schema_cls is DummyModel
        return schema_cls(name=text.split(":")[-1].strip())

    monkeypatch.setattr("mosaicx.api.extraction.load_schema_model", fake_load_schema_model)
    monkeypatch.setattr("mosaicx.api.extraction.extract_text_from_document", fake_extract_text)
    monkeypatch.setattr("mosaicx.api.extraction.extract_structured_data", fake_extract_structured)


def test_extract_document_accepts_paths(tmp_path: Path, patch_extraction: None, dummy_schema_file: Path) -> None:
    doc_path = tmp_path / "report.pdf"
    doc_path.write_text("content", encoding="utf-8")

    result = extract_document(doc_path, dummy_schema_file)
    assert result.record.name == "Alice"

    out_path = tmp_path / "out.json"
    result.write_json(out_path)
    payload = json.loads(out_path.read_text())
    assert payload == {"name": "Alice"}


def test_summarize_reports_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("Example", encoding="utf-8")

    def fake_load_reports(paths):
        assert paths == [doc]
        return ["doc"]

    def fake_summarize(_docs, **_kwargs):
        from mosaicx.summarizer import PatientSummary, PatientHeader

        return PatientSummary(
            patient=PatientHeader(patient_id="demo", last_updated="2025-10-11T10:00:00Z"),
            timeline=[],
            overall="ok",
        )

    saved = {}

    def fake_artifacts(ps, *, artifacts, json_path, pdf_path, patient_id, emit_messages):
        saved["artifacts"] = tuple(artifacts)
        saved["json_path"] = json_path
        saved["pdf_path"] = pdf_path
        return {fmt: Path("dummy" + fmt) for fmt in artifacts}

    monkeypatch.setattr("mosaicx.api.summary.load_reports", fake_load_reports)
    monkeypatch.setattr("mosaicx.api.summary.summarize_with_llm", fake_summarize)
    monkeypatch.setattr("mosaicx.api.summary.save_summary_artifacts", fake_artifacts)

    summarize_reports([
        str(doc),
    ], patient_id="demo", artifacts=["json", "pdf"], json_path=tmp_path / "custom.json", pdf_path=tmp_path / "custom.pdf")

    assert saved["artifacts"] == ("json", "pdf")
    assert saved["json_path"].name == "custom.json"
    assert saved["pdf_path"].name == "custom.pdf"

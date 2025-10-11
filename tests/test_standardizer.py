from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json
import pytest
from pydantic import BaseModel

from mosaicx.standardizer import StandardizeResult, save_standardize_artifacts


class DemoSchema(BaseModel):
    patient_id: str
    finding: str
    measurements: list[str]


def _sample_result(tmp_path: Path) -> StandardizeResult:
    record = DemoSchema(
        patient_id="P001",
        finding="No acute findings.",
        measurements=["CT 2025-01-02 normal"],
    )
    return StandardizeResult(
        record=record,
        schema_path=tmp_path / "demo_schema.py",
        document_path=tmp_path / "input.pdf",
        schema_name="DemoSchema",
        model_name="demo-model",
        generated_at=datetime(2025, 1, 2, 12, 30, tzinfo=timezone.utc),
        extracted_text="Sample text",
    )


def test_save_standardize_artifacts_json_and_txt(tmp_path: Path) -> None:
    result = _sample_result(tmp_path)
    result.narrative = "Patient is stable with no acute findings."
    saved = save_standardize_artifacts(
        result,
        artifacts=("json", "txt"),
        json_path=None,
        pdf_path=None,
        text_path=None,
        docx_path=None,
        out_dir=tmp_path,
        emit_messages=False,
    )

    assert "json" in saved
    assert "txt" in saved

    json_path = saved["json"]
    txt_path = saved["txt"]

    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["patient_id"] == "P001"
    assert payload["measurements"] == ["CT 2025-01-02 normal"]

    assert txt_path.exists()
    text_content = txt_path.read_text(encoding="utf-8")
    assert "MOSAICX Structured Report" in text_content
    assert "FINDINGS:" in text_content
    assert "Patient is stable with no acute findings." in text_content
    assert "Patient Id" in text_content
    assert "P001" in text_content


def test_save_standardize_artifacts_respects_invalid(tmp_path: Path) -> None:
    result = _sample_result(tmp_path)
    with pytest.raises(ValueError):
        save_standardize_artifacts(
            result,
            artifacts=("unsupported",),
            json_path=None,
            pdf_path=None,
            text_path=None,
            docx_path=None,
            out_dir=tmp_path,
            emit_messages=False,
        )

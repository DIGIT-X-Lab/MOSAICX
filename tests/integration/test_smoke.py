from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from mosaicx.api import extract_document, summarize_reports

SAMPLE_SUMMARY_REPORTS = [
    Path("tests/datasets/summarize/P001_CT_2025-08-01.pdf"),
    Path("tests/datasets/summarize/P001_CT_2025-09-10.pdf"),
]
SAMPLE_EXTRACT_PDF = Path("tests/datasets/extract/sample_patient_vitals.pdf")
SCHEMA_PATH = Path("mosaicx/schema/templates/python/patient_identity.py")


@pytest.mark.integration
def test_api_summarize_reports(tmp_path: Path) -> None:
    json_target = tmp_path / "api_summary.json"
    pdf_target = tmp_path / "api_summary.pdf"

    summary = summarize_reports(
        SAMPLE_SUMMARY_REPORTS,
        patient_id="P001",
        artifacts=["json", "pdf"],
        json_path=json_target,
        pdf_path=pdf_target,
    )

    assert summary.overall
    assert json_target.exists()
    assert pdf_target.exists()


@pytest.mark.integration
def test_api_extract_document(tmp_path: Path) -> None:
    result = extract_document(SAMPLE_EXTRACT_PDF, SCHEMA_PATH)
    json_path = tmp_path / "api_extraction.json"
    result.write_json(json_path)

    data = result.to_dict()
    assert isinstance(data, dict)
    assert data  # non-empty payload
    assert json_path.exists()


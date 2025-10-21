from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from mosaicx.summarizer import (
    CriticalEvent,
    PatientHeader,
    PatientSummary,
    save_summary_artifacts,
    summarize_reports_to_terminal_and_json,
)


@pytest.fixture
def sample_summary() -> PatientSummary:
    return PatientSummary(
        patient=PatientHeader(patient_id="PX-10", last_updated="2025-10-11T11:00:00Z"),
        timeline=[CriticalEvent(date="2025-10-10", source="report.txt", note="Stable")],
        overall="Stable condition",
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_save_json_by_default(tmp_output_dir: Path, sample_summary: PatientSummary, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Path] = {}

    def fake_write_json(ps: PatientSummary, path: Path) -> None:
        recorded["json"] = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("mosaicx.summarizer.write_summary_json", fake_write_json)
    monkeypatch.setattr("mosaicx.summarizer.write_summary_pdf", lambda *_args, **_kwargs: None)

    saved = save_summary_artifacts(
        sample_summary,
        artifacts=["json"],
        json_path=None,
        pdf_path=None,
        patient_id="PX-10",
        emit_messages=False,
    )

    assert "json" in saved
    assert saved["json"].suffix == ".json"
    assert saved["json"].exists()


def test_save_json_and_pdf_custom_paths(tmp_output_dir: Path, sample_summary: PatientSummary, monkeypatch: pytest.MonkeyPatch) -> None:
    written = {"json": None, "pdf": None}

    def fake_write_json(_ps: PatientSummary, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        written["json"] = path

    def fake_write_pdf(_ps: PatientSummary, path: Path, **_kwargs) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n")
        written["pdf"] = path

    monkeypatch.setattr("mosaicx.summarizer.write_summary_json", fake_write_json)
    monkeypatch.setattr("mosaicx.summarizer.write_summary_pdf", fake_write_pdf)

    custom_json = tmp_output_dir / "out" / "summary.json"
    custom_pdf = tmp_output_dir / "out" / "summary.pdf"

    saved = save_summary_artifacts(
        sample_summary,
        artifacts=["json", "pdf"],
        json_path=custom_json,
        pdf_path=custom_pdf,
        patient_id="PX-10",
        emit_messages=False,
    )

    assert saved["json"] == custom_json
    assert saved["pdf"] == custom_pdf
    assert custom_json.exists()
    assert custom_pdf.exists()


def test_summarize_reports_to_terminal_and_json_invokes_artifact_helper(tmp_output_dir: Path, monkeypatch: pytest.MonkeyPatch, sample_summary: PatientSummary) -> None:
    calls: dict[str, Iterable[str]] = {}

    def fake_save(
        ps: PatientSummary,
        *,
        artifacts: Iterable[str],
        json_path,
        pdf_path,
        patient_id,
        model_name,
        emit_messages,
    ):
        calls["artifacts"] = tuple(artifacts)
        calls["model"] = model_name
        return {asset: Path("dummy" + asset) for asset in artifacts}

    monkeypatch.setattr("mosaicx.summarizer.save_summary_artifacts", fake_save)
    monkeypatch.setattr("mosaicx.summarizer.render_summary_rich", lambda *_: None)
    monkeypatch.setattr("mosaicx.summarizer.summarize_with_llm", lambda *_args, **_kwargs: sample_summary)
    monkeypatch.setattr("mosaicx.summarizer.load_reports", lambda paths: ["doc"])

    result = summarize_reports_to_terminal_and_json(
        paths=[Path("tests/datasets/extract/sample_patient_vitals.pdf")],
        patient_id="PZ",
        model="demo-model",
        base_url=None,
        api_key=None,
        temperature=0.2,
        artifacts=("json", "pdf"),
        json_out=None,
        pdf_out=None,
    )

    assert result == sample_summary
    assert calls["artifacts"] == ("json", "pdf")
    assert calls["model"] == "demo-model"

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner


def test_apply_extraction_contract_basic_semantics():
    from mosaicx.pipelines.extraction import apply_extraction_contract

    payload = {
        "extracted": {
            "bp": "128/82",
            "missing_field": None,
            "age": 52,
        }
    }

    result = apply_extraction_contract(payload, source_text="Vitals: BP 128/82. Age 52.")
    contract = result["_extraction_contract"]
    assert contract["version"] == "1.0"
    assert set(contract["critical_fields"]) == {"bp", "missing_field", "age"}

    by_field = {row["field"]: row for row in contract["field_results"]}
    assert by_field["bp"]["status"] == "supported"
    assert by_field["bp"]["grounded"] is True
    assert "128/82" in str(by_field["bp"]["evidence"])

    assert by_field["missing_field"]["status"] == "insufficient_evidence"
    assert by_field["missing_field"]["grounded"] is False
    assert by_field["missing_field"]["confidence"] == 0.0

    assert by_field["age"]["status"] == "supported"
    assert by_field["age"]["grounded"] is True

    assert contract["summary"]["supported"] == 2
    assert contract["summary"]["insufficient_evidence"] == 1


def test_sdk_extract_attaches_extraction_contract(monkeypatch):
    import mosaicx.sdk as sdk

    monkeypatch.setattr(sdk, "_ensure_configured", lambda: None)

    class _FakeExtractor:
        def __init__(self, *args, **kwargs):
            self._last_metrics = None

        def __call__(self, document_text: str):
            return SimpleNamespace(extracted={"bp": "128/82"})

    monkeypatch.setattr("mosaicx.pipelines.extraction.DocumentExtractor", _FakeExtractor)

    result = sdk.extract(text="Patient BP is 128/82.")
    assert "_extraction_contract" in result
    contract = result["_extraction_contract"]
    assert contract["summary"]["supported"] >= 1
    assert any(r["field"] == "bp" and r["status"] == "supported" for r in contract["field_results"])


def test_cli_extract_output_includes_extraction_contract(monkeypatch, tmp_path):
    from mosaicx.cli import cli

    monkeypatch.setattr("mosaicx.cli._check_api_key", lambda: None)
    monkeypatch.setattr("mosaicx.cli._configure_dspy", lambda: None)

    fake_doc = SimpleNamespace(
        text="Vitals: BP 128/82.",
        quality_warning=False,
        is_empty=False,
        format="txt",
        char_count=20,
        ocr_engine_used=None,
        ocr_confidence=None,
    )
    monkeypatch.setattr("mosaicx.cli._load_doc_with_config", lambda _path: fake_doc)

    class _FakeExtractor:
        def __init__(self, *args, **kwargs):
            self._last_metrics = None

        def __call__(self, document_text: str):
            return SimpleNamespace(extracted={"bp": "128/82"})

    monkeypatch.setattr("mosaicx.pipelines.extraction.DocumentExtractor", _FakeExtractor)

    doc_path = tmp_path / "report.txt"
    doc_path.write_text("dummy", encoding="utf-8")
    out_path = tmp_path / "result.json"

    runner = CliRunner()
    res = runner.invoke(cli, ["extract", "--document", str(doc_path), "-o", str(out_path)])
    assert res.exit_code == 0, res.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "_extraction_contract" in payload
    assert payload["_extraction_contract"]["summary"]["supported"] >= 1


def test_mcp_extract_document_includes_extraction_contract(monkeypatch):
    import mosaicx.mcp_server as mcp_mod

    monkeypatch.setattr(mcp_mod, "_ensure_dspy", lambda: None)

    class _FakeExtractor:
        def __init__(self, *args, **kwargs):
            self._last_metrics = None

        def __call__(self, document_text: str):
            return SimpleNamespace(extracted={"bp": "128/82"})

    monkeypatch.setattr("mosaicx.pipelines.extraction.DocumentExtractor", _FakeExtractor)

    raw = mcp_mod.extract_document(document_text="Patient BP is 128/82.")
    payload = json.loads(raw)
    assert "_extraction_contract" in payload
    assert payload["_extraction_contract"]["summary"]["supported"] >= 1

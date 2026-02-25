# tests/test_extraction_pipeline.py
"""Tests for the extraction pipeline."""
import pytest
from pydantic import BaseModel


class TestInferSchemaSignature:
    def test_infer_schema_signature_fields(self):
        from mosaicx.pipelines.extraction import InferSchemaFromDocument
        assert "document_text" in InferSchemaFromDocument.input_fields
        assert "schema_spec" in InferSchemaFromDocument.output_fields


class TestDocumentExtractorModule:
    def test_auto_mode_has_infer_and_extract(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        extractor = DocumentExtractor()
        assert hasattr(extractor, "infer_schema")

    def test_schema_mode_has_extract_custom(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class CustomReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=CustomReport)
        assert hasattr(extractor, "extract_custom")
        assert not hasattr(extractor, "infer_schema")

    def test_schema_mode_coerces_model_instance_output(self, monkeypatch):
        from enum import Enum

        from pydantic import BaseModel

        from mosaicx.pipelines.extraction import DocumentExtractor

        class OptionalSeverity(str, Enum):
            Mild = "Mild"
            None_ = "None"

        class Finding(BaseModel):
            level: str | None = None
            severity: OptionalSeverity | None = None

        class CustomReport(BaseModel):
            finding: Finding | None = None

        extractor = DocumentExtractor(output_schema=CustomReport)

        extracted_instance = CustomReport(
            finding=Finding(level="C2‑3", severity=None)
        )

        class _DummyPrediction:
            extracted = extracted_instance

        monkeypatch.setattr(
            extractor,
            "extract_custom",
            lambda document_text: _DummyPrediction(),
        )

        result = extractor.forward("synthetic report text")
        assert result.extracted.finding is not None
        assert result.extracted.finding.level == "C2-C3"
        assert result.extracted.finding.severity == OptionalSeverity.None_


class TestConvenienceFunctions:
    def test_extract_with_schema_function_exists(self):
        from mosaicx.pipelines.extraction import extract_with_schema
        assert callable(extract_with_schema)

    def test_extract_with_mode_function_exists(self):
        from mosaicx.pipelines.extraction import extract_with_mode
        assert callable(extract_with_mode)


class TestExtractionPlannerRouting:
    def test_plan_routes_fallback_assigns_easy_and_hard_sections(self, monkeypatch):
        from mosaicx.pipelines.extraction import (
            _plan_section_routes_with_react,
            _split_document_sections,
        )

        # Force deterministic fallback path so route selection is stable in tests.
        monkeypatch.setattr(
            "mosaicx.runtime_env.import_dspy",
            lambda: (_ for _ in ()).throw(RuntimeError("no dspy")),
        )

        doc = (
            "Clinical Information:\n"
            "Neck pain.\n\n"
            "Findings:\n"
            "C3-4 bulge with 12 mm lesion and 16 mm follow-up measurement. "
            "Multilevel narrowing. " * 8
        )
        sections = _split_document_sections(doc)
        routes, meta = _plan_section_routes_with_react(
            schema_name="MRICervicalSpineV3",
            sections=sections,
        )

        assert meta["planner"] == "deterministic_fallback"
        by_section = {r["section"].lower(): r for r in routes}
        assert by_section["clinical information"]["strategy"] in {
            "deterministic",
            "constrained_extract",
        }
        assert by_section["findings"]["strategy"] in {"heavy_extract", "repair"}

    def test_plan_extraction_document_text_returns_diagnostics(self, monkeypatch):
        from mosaicx.pipelines.extraction import _plan_extraction_document_text

        monkeypatch.setattr(
            "mosaicx.runtime_env.import_dspy",
            lambda: (_ for _ in ()).throw(RuntimeError("no dspy")),
        )

        doc = (
            "Clinical Information:\n"
            "Neck pain.\n\n"
            "Findings:\n"
            "Bulge at C3-4 with foraminal narrowing.\n\n"
            "Impression:\n"
            "Mild progression."
        )
        planned_text, diag = _plan_extraction_document_text(
            document_text=doc,
            schema_name="MRICervicalSpineV3",
        )

        assert isinstance(planned_text, str) and planned_text.strip()
        assert isinstance(diag.get("routes"), list) and diag["routes"]
        assert "strategy_counts" in diag
        assert diag.get("planned_chars", 0) > 0
        assert diag.get("original_chars", 0) >= diag.get("planned_chars", 0)

    def test_schema_mode_uses_planned_text_and_exposes_planner(self, monkeypatch):
        from mosaicx.pipelines.extraction import DocumentExtractor

        class CustomReport(BaseModel):
            summary: str

        extractor = DocumentExtractor(output_schema=CustomReport)

        planned = "Findings:\nRouted concise context."
        planner_diag = {
            "planner": "react",
            "react_used": True,
            "routes": [{"section": "Findings", "strategy": "constrained_extract"}],
            "strategy_counts": {
                "deterministic": 0,
                "constrained_extract": 1,
                "heavy_extract": 0,
                "repair": 0,
            },
            "original_chars": 120,
            "planned_chars": 40,
            "compression_ratio": 0.33,
            "sections": [{"name": "findings", "title": "Findings", "chars": 120}],
        }

        monkeypatch.setattr(
            "mosaicx.pipelines.extraction._plan_extraction_document_text",
            lambda *, document_text, schema_name: (planned, planner_diag),
        )

        captured: dict[str, str] = {}

        class _DummyPrediction:
            extracted = CustomReport(summary="stable")

        def _fake_extract_custom(*, document_text: str):
            captured["document_text"] = document_text
            return _DummyPrediction()

        monkeypatch.setattr(extractor, "extract_custom", _fake_extract_custom)

        result = extractor.forward("Original long source text that should be routed.")
        assert captured["document_text"] == planned
        assert getattr(result, "planner") == planner_diag
        assert extractor._last_planner == planner_diag

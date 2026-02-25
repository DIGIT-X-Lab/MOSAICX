# tests/test_extraction_pipeline.py
"""Tests for the extraction pipeline."""
import pytest


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

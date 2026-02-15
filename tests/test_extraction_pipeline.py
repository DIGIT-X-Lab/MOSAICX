# tests/test_extraction_pipeline.py
"""Tests for the generic document extraction pipeline."""

import pytest


class TestExtractionOutputModels:
    """Test the Pydantic output models."""

    def test_demographics_model(self):
        from mosaicx.pipelines.extraction import Demographics
        d = Demographics(patient_id="P001", name="John", age=65, gender="Male")
        assert d.patient_id == "P001"
        assert d.age == 65

    def test_finding_model(self):
        from mosaicx.pipelines.extraction import Finding
        f = Finding(description="5mm nodule", location="RUL", severity="mild")
        assert f.description == "5mm nodule"

    def test_diagnosis_model(self):
        from mosaicx.pipelines.extraction import Diagnosis
        d = Diagnosis(name="Pneumonia", icd10_code="J18.9")
        assert d.name == "Pneumonia"


class TestExtractionSignatures:
    """Test DSPy signatures have correct fields."""

    def test_extract_demographics_signature(self):
        from mosaicx.pipelines.extraction import ExtractDemographics
        assert "document_text" in ExtractDemographics.input_fields
        assert "demographics" in ExtractDemographics.output_fields

    def test_extract_findings_signature(self):
        from mosaicx.pipelines.extraction import ExtractFindings
        assert "document_text" in ExtractFindings.input_fields
        assert "findings" in ExtractFindings.output_fields

    def test_extract_diagnoses_signature(self):
        from mosaicx.pipelines.extraction import ExtractDiagnoses
        assert "document_text" in ExtractDiagnoses.input_fields
        assert "diagnoses" in ExtractDiagnoses.output_fields


class TestDocumentExtractorModule:
    """Test the DocumentExtractor DSPy module."""

    def test_module_has_submodules(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        extractor = DocumentExtractor()
        assert hasattr(extractor, "extract_demographics")
        assert hasattr(extractor, "extract_findings")
        assert hasattr(extractor, "extract_diagnoses")

    def test_module_accepts_custom_schema(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class CustomReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=CustomReport)
        assert hasattr(extractor, "extract_custom")

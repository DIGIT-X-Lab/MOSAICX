"""Tests for the completeness evaluator."""
import pytest
from pydantic import BaseModel
from typing import Optional

class SampleReport(BaseModel):
    indication: str
    findings: list[str]
    impression: str
    technique: Optional[str] = None

class TestFieldCoverage:
    def test_all_fields_populated(self):
        from mosaicx.evaluation.completeness import field_coverage
        report = SampleReport(indication="Cough", findings=["nodule"], impression="Follow-up", technique="CT")
        score = field_coverage(report)
        assert score == 1.0

    def test_optional_field_missing(self):
        from mosaicx.evaluation.completeness import field_coverage
        report = SampleReport(indication="Cough", findings=["nodule"], impression="Follow-up")
        score = field_coverage(report)
        assert 0.5 < score < 1.0

    def test_empty_list_penalized(self):
        from mosaicx.evaluation.completeness import field_coverage
        report = SampleReport(indication="Cough", findings=[], impression="Normal")
        score = field_coverage(report)
        assert score < 1.0

    def test_empty_string_penalized(self):
        from mosaicx.evaluation.completeness import field_coverage
        report = SampleReport(indication="", findings=["nodule"], impression="Normal")
        score = field_coverage(report)
        assert score < 1.0

class TestInformationDensity:
    def test_density_ratio(self):
        from mosaicx.evaluation.completeness import information_density
        source = "This is a very long report with many findings and detailed descriptions."
        structured = SampleReport(indication="Report", findings=["finding1"], impression="Normal")
        density = information_density(source, structured)
        assert 0.0 <= density <= 1.0

    def test_density_empty_source(self):
        from mosaicx.evaluation.completeness import information_density
        structured = SampleReport(indication="x", findings=["x"], impression="x")
        density = information_density("", structured)
        assert density == 0.0

class TestCompletenessScore:
    def test_overall_score_structure(self):
        from mosaicx.evaluation.completeness import compute_completeness
        report = SampleReport(indication="Cough", findings=["5mm nodule"], impression="Follow-up", technique="CT")
        result = compute_completeness(report, source_text="Patient presents with cough. CT chest. 5mm nodule in RUL.")
        assert "overall" in result
        assert "field_coverage" in result
        assert "information_density" in result
        assert 0.0 <= result["overall"] <= 1.0

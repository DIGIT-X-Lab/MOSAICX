# tests/test_radiology_pipeline.py
"""Tests for the radiology report structurer pipeline."""

import pytest


class TestRadReportBaseModels:
    def test_measurement_construction(self):
        from mosaicx.schemas.radreport.base import Measurement
        m = Measurement(value=5.2, unit="mm", dimension="diameter")
        assert m.value == 5.2
        assert m.unit == "mm"

    def test_finding_construction(self):
        from mosaicx.schemas.radreport.base import RadReportFinding
        f = RadReportFinding(
            anatomy="right upper lobe",
            observation="nodule",
            description="5mm ground glass nodule",
        )
        assert f.anatomy == "right upper lobe"
        assert f.radlex_id is None

    def test_impression_item(self):
        from mosaicx.schemas.radreport.base import ImpressionItem
        imp = ImpressionItem(
            statement="Pulmonary nodule, recommend CT follow-up in 12 months.",
            category="Lung-RADS 3",
            actionable=True,
        )
        assert imp.actionable is True

    def test_change_type(self):
        from mosaicx.schemas.radreport.base import ChangeType
        c = ChangeType(status="increased", prior_date="2025-06-15")
        assert c.status == "increased"

    def test_report_sections(self):
        from mosaicx.schemas.radreport.base import ReportSections
        s = ReportSections(indication="Cough", findings="5mm nodule RUL", impression="Follow-up.")
        assert s.indication == "Cough"
        assert s.comparison == ""  # default empty


class TestRadiologyPipelineSignatures:
    def test_classify_exam_type_signature(self):
        from mosaicx.pipelines.radiology import ClassifyExamType
        assert "report_header" in ClassifyExamType.input_fields
        assert "exam_type" in ClassifyExamType.output_fields

    def test_parse_sections_signature(self):
        from mosaicx.pipelines.radiology import ParseReportSections
        assert "report_text" in ParseReportSections.input_fields
        assert "sections" in ParseReportSections.output_fields

    def test_extract_findings_signature(self):
        from mosaicx.pipelines.radiology import ExtractRadFindings
        assert "findings_text" in ExtractRadFindings.input_fields
        assert "findings" in ExtractRadFindings.output_fields

    def test_extract_impression_signature(self):
        from mosaicx.pipelines.radiology import ExtractImpression
        assert "impression_text" in ExtractImpression.input_fields
        assert "impressions" in ExtractImpression.output_fields


class TestRadiologyStructurerModule:
    def test_module_has_submodules(self):
        from mosaicx.pipelines.radiology import RadiologyReportStructurer
        pipeline = RadiologyReportStructurer()
        assert hasattr(pipeline, "classify_exam")
        assert hasattr(pipeline, "parse_sections")
        assert hasattr(pipeline, "extract_findings")
        assert hasattr(pipeline, "extract_impression")

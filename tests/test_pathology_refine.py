# tests/test_pathology_refine.py
"""Tests for the optional dspy.Refine wrappers on the pathology pipeline."""

from __future__ import annotations


class TestPathologyPipelineRefineWiring:
    def test_pipeline_has_extract_findings_module(self):
        """PathologyReportStructurer always has extract_findings."""
        from mosaicx.pipelines.pathology import PathologyReportStructurer

        pipeline = PathologyReportStructurer()
        assert hasattr(pipeline, "extract_findings")

    def test_pipeline_has_extract_diagnosis_module(self):
        from mosaicx.pipelines.pathology import PathologyReportStructurer

        pipeline = PathologyReportStructurer()
        assert hasattr(pipeline, "extract_diagnosis")

# tests/test_radiology_refine.py
"""Tests for the optional dspy.Refine wrappers on the radiology pipeline."""

from __future__ import annotations

import pytest


class TestRadiologyRefineConfig:
    def test_config_has_use_refine_flag(self):
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig(api_key="test")
        assert hasattr(cfg, "use_refine")
        assert isinstance(cfg.use_refine, bool)

    def test_use_refine_defaults_false(self):
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig(api_key="test")
        assert cfg.use_refine is False


class TestRadiologyPipelineRefineWiring:
    def test_pipeline_has_extract_findings_module(self):
        """RadiologyReportStructurer always has extract_findings, whether Refine or ChainOfThought."""
        from mosaicx.pipelines.radiology import RadiologyReportStructurer

        pipeline = RadiologyReportStructurer()
        assert hasattr(pipeline, "extract_findings")

    def test_pipeline_has_extract_impression_module(self):
        from mosaicx.pipelines.radiology import RadiologyReportStructurer

        pipeline = RadiologyReportStructurer()
        assert hasattr(pipeline, "extract_impression")

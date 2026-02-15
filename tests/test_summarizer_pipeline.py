# tests/test_summarizer_pipeline.py
"""Tests for the report summarizer pipeline."""

import pytest


class TestSummarizerModels:
    def test_timeline_event_model(self):
        from mosaicx.pipelines.summarizer import TimelineEvent
        ev = TimelineEvent(
            date="2025-06-15",
            exam_type="CT chest",
            key_finding="5mm RUL nodule, new",
            clinical_context="Cough for 3 weeks",
        )
        assert ev.exam_type == "CT chest"
        assert ev.date == "2025-06-15"

    def test_timeline_event_optional_fields(self):
        from mosaicx.pipelines.summarizer import TimelineEvent
        ev = TimelineEvent(date="2025-01-01", exam_type="XR", key_finding="Normal")
        assert ev.clinical_context is None


class TestSummarizerSignatures:
    def test_extract_timeline_event_signature(self):
        from mosaicx.pipelines.summarizer import ExtractTimelineEvent
        assert "report_text" in ExtractTimelineEvent.input_fields
        assert "event" in ExtractTimelineEvent.output_fields

    def test_synthesize_timeline_signature(self):
        from mosaicx.pipelines.summarizer import SynthesizeTimeline
        assert "events_json" in SynthesizeTimeline.input_fields
        assert "narrative" in SynthesizeTimeline.output_fields


class TestSummarizerModule:
    def test_module_has_submodules(self):
        from mosaicx.pipelines.summarizer import ReportSummarizer
        s = ReportSummarizer()
        assert hasattr(s, "extract_event")
        assert hasattr(s, "synthesize")

# tests/test_sdk_envelope.py
"""Tests for _mosaicx envelope attachment in SDK functions.

These tests verify the envelope wiring logic without calling LLMs.
We test that build_envelope() produces the right structure when called
with the arguments that each SDK function would pass, and that the
resulting dict can be attached to a result dict alongside existing keys.
"""
from __future__ import annotations


class TestExtractEnvelopeAttachment:
    """Verify envelope attachment pattern for _extract_single_text."""

    def test_envelope_attaches_to_extract_result(self):
        """Simulates what _extract_single_text does: attach _mosaicx to output."""
        from mosaicx.envelope import build_envelope

        # Simulate a typical extraction output dict
        output_data = {
            "findings": [{"text": "Normal heart size"}],
            "impression": "No acute findings",
        }

        # Attach envelope as _extract_single_text would
        output_data["_mosaicx"] = build_envelope(
            pipeline="radiology",
            template="chest_ct",
        )

        assert "_mosaicx" in output_data
        assert output_data["_mosaicx"]["pipeline"] == "radiology"
        assert output_data["_mosaicx"]["template"] == "chest_ct"

    def test_envelope_coexists_with_metrics(self):
        """_mosaicx and _metrics should both be present (backward compat)."""
        from mosaicx.envelope import build_envelope

        output_data = {
            "findings": [],
            "_metrics": {
                "total_duration_s": 2.5,
                "total_tokens": 1000,
                "steps": [],
            },
        }

        output_data["_mosaicx"] = build_envelope(
            pipeline="radiology",
            duration_s=2.5,
            tokens={"input": 600, "output": 400},
        )

        # Both keys present
        assert "_metrics" in output_data
        assert "_mosaicx" in output_data
        # Envelope captures duration from metrics
        assert output_data["_mosaicx"]["duration_s"] == 2.5
        assert output_data["_mosaicx"]["tokens"] == {
            "input": 600,
            "output": 400,
        }

    def test_envelope_with_mode_extraction(self):
        """Mode-based extraction sets pipeline to the mode name, no template."""
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        assert env["pipeline"] == "radiology"
        assert env["template"] is None

    def test_envelope_with_template_extraction(self):
        """Template-based extraction sets both pipeline and template."""
        from mosaicx.envelope import build_envelope

        env = build_envelope(
            pipeline="radiology",
            template="chest_ct",
        )
        assert env["pipeline"] == "radiology"
        assert env["template"] == "chest_ct"

    def test_envelope_with_auto_mode(self):
        """Auto mode uses 'extraction' as pipeline name."""
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="extraction")
        assert env["pipeline"] == "extraction"
        assert env["template"] is None

    def test_envelope_does_not_overwrite_existing_keys(self):
        """Envelope attachment should not disturb other output keys."""
        from mosaicx.envelope import build_envelope

        output_data = {
            "extracted": {"field": "value"},
            "completeness": {"score": 0.85},
            "_metrics": {"total_duration_s": 1.0},
        }

        output_data["_mosaicx"] = build_envelope(pipeline="extraction")

        # All original keys still present
        assert "extracted" in output_data
        assert "completeness" in output_data
        assert "_metrics" in output_data
        assert "_mosaicx" in output_data


class TestDeidentifyEnvelopeAttachment:
    """Verify envelope attachment pattern for _deidentify_single_text."""

    def test_envelope_attaches_to_deidentify_result(self):
        """Simulates what _deidentify_single_text does."""
        from mosaicx.envelope import build_envelope

        output_data = {"redacted_text": "[REDACTED] presented with chest pain."}

        output_data["_mosaicx"] = build_envelope(pipeline="deidentify")

        assert "_mosaicx" in output_data
        assert output_data["_mosaicx"]["pipeline"] == "deidentify"
        assert output_data["_mosaicx"]["template"] is None

    def test_deidentify_envelope_no_template(self):
        """Deidentify pipeline never has a template."""
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="deidentify")
        assert env["pipeline"] == "deidentify"
        assert env["template"] is None
        assert env["template_version"] is None


class TestSummarizeEnvelopeAttachment:
    """Verify envelope attachment pattern for summarize."""

    def test_envelope_attaches_to_summarize_result(self):
        """Simulates what summarize() does."""
        from mosaicx.envelope import build_envelope

        output_data = {
            "narrative": "Patient had two visits...",
            "events": [{"date": "2024-01-15", "description": "CT scan"}],
        }

        output_data["_mosaicx"] = build_envelope(pipeline="summarize")

        assert "_mosaicx" in output_data
        assert output_data["_mosaicx"]["pipeline"] == "summarize"
        assert output_data["_mosaicx"]["template"] is None

    def test_summarize_envelope_coexists_with_document(self):
        """_mosaicx should coexist with _document metadata."""
        from mosaicx.envelope import build_envelope

        output_data = {
            "narrative": "Summary text",
            "events": [],
            "_document": [
                {"file": "report1.pdf", "page_count": 2},
                {"file": "report2.pdf", "page_count": 1},
            ],
        }

        output_data["_mosaicx"] = build_envelope(pipeline="summarize")

        assert "_document" in output_data
        assert "_mosaicx" in output_data


class TestEnvelopeWithMetricsData:
    """Test that metrics data flows correctly into the envelope."""

    def test_duration_from_pipeline_metrics(self):
        """Envelope should accept duration_s from PipelineMetrics."""
        from mosaicx.metrics import PipelineMetrics, StepMetric

        metrics = PipelineMetrics(
            steps=[
                StepMetric("step1", duration_s=1.0, input_tokens=100, output_tokens=50),
                StepMetric("step2", duration_s=2.0, input_tokens=200, output_tokens=100),
            ]
        )

        from mosaicx.envelope import build_envelope

        env = build_envelope(
            pipeline="radiology",
            duration_s=metrics.total_duration_s,
            tokens={
                "input": metrics.total_input_tokens,
                "output": metrics.total_output_tokens,
            },
        )

        assert env["duration_s"] == 3.0
        assert env["tokens"] == {"input": 300, "output": 150}

    def test_none_metrics_uses_defaults(self):
        """When metrics are None, envelope should use defaults."""
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        assert env["duration_s"] is None
        assert env["tokens"] == {"input": 0, "output": 0}

"""Tests for the verification engine orchestrator."""

from __future__ import annotations

import pytest


class TestVerifyEngine:
    def test_quick_level_uses_deterministic(self):
        from mosaicx.verify.engine import verify

        extraction = {"findings": []}
        source = "Normal exam."
        report = verify(extraction=extraction, source_text=source, level="quick")
        assert report.level == "deterministic"

    def test_invalid_level_raises(self):
        from mosaicx.verify.engine import verify

        with pytest.raises(ValueError, match="Unknown verification level"):
            verify(extraction={}, source_text="", level="invalid")

    def test_claim_based_verify(self):
        from mosaicx.verify.engine import verify

        report = verify(
            claim="EF was 45%", source_text="EF estimated at 45%.", level="quick"
        )
        assert report.verdict == "verified"

    def test_must_provide_extraction_or_claim(self):
        from mosaicx.verify.engine import verify

        with pytest.raises(ValueError, match="Must provide"):
            verify(source_text="text", level="quick")

    def test_standard_level_returns_report(self):
        from mosaicx.verify.engine import verify

        report = verify(
            extraction={"findings": []}, source_text="Normal.", level="standard"
        )
        assert report.verdict == "verified"

    def test_thorough_level_returns_report(self):
        from mosaicx.verify.engine import verify

        report = verify(
            extraction={"findings": []}, source_text="Normal.", level="thorough"
        )
        assert report.verdict == "verified"

    def test_claim_with_missing_numbers(self):
        from mosaicx.verify.engine import verify

        report = verify(
            claim="EF was 65%",
            source_text="Left ventricular function is normal.",
            level="quick",
        )
        # 65 not in source text, should not be verified
        assert report.verdict != "verified"
        assert any(i.type == "value_not_found" for i in report.issues)

    def test_claim_with_low_word_overlap(self):
        from mosaicx.verify.engine import verify

        report = verify(
            claim="Tumor is malignant grade III",
            source_text="The weather is sunny today.",
            level="quick",
        )
        assert report.verdict == "insufficient_evidence"

    def test_extraction_with_measurement_issues(self):
        from mosaicx.verify.engine import verify

        extraction = {
            "findings": [{"measurement": {"value": 99.0, "unit": "mm"}}]
        }
        source = "No such number in source."
        report = verify(extraction=extraction, source_text=source, level="quick")
        assert report.verdict == "partially_supported"
        assert len(report.issues) > 0

    def test_claim_verified_confidence_capped(self):
        from mosaicx.verify.engine import verify

        report = verify(
            claim="EF 45%",
            source_text="EF estimated at 45% by echocardiography.",
            level="quick",
        )
        assert report.verdict == "verified"
        assert report.confidence <= 0.95

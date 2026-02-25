"""Tests for the verification engine orchestrator."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from mosaicx.config import MosaicxConfig
from mosaicx.verify.models import VerificationReport


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

        # Empty extraction → insufficient_evidence (no verifiable fields)
        report = verify(
            extraction={"findings": []}, source_text="Normal.", level="standard"
        )
        assert report.verdict == "insufficient_evidence"

    def test_thorough_level_returns_report(self):
        from mosaicx.verify.engine import verify

        # Empty extraction → insufficient_evidence (no verifiable fields)
        report = verify(
            extraction={"findings": []}, source_text="Normal.", level="thorough"
        )
        assert report.verdict == "insufficient_evidence"

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


class TestSmartRouter:
    """Tests for _should_use_rlm_audit smart routing."""

    def test_short_doc_few_fields_skips_rlm(self):
        from mosaicx.verify.engine import _should_use_rlm_audit

        extraction = {"diagnosis": "Adenocarcinoma", "grade": "II", "margin": "negative"}
        source = "Short report. " * 20  # ~300 chars
        base_report = VerificationReport(
            verdict="verified", confidence=0.8, level="deterministic"
        )
        use_rlm, reason = _should_use_rlm_audit(extraction, source, base_report)
        assert use_rlm is False
        assert "short_doc" in reason
        assert "few_fields" in reason

    def test_local_model_skips_rlm(self, monkeypatch):
        from mosaicx.verify.engine import _should_use_rlm_audit

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(api_base="http://localhost:11434/v1"),
        )
        # Large doc with many fields — would normally trigger RLM
        extraction = {f"field_{i}": f"value number {i} with enough length" for i in range(20)}
        source = "A" * 10000
        base_report = VerificationReport(
            verdict="partially_supported", confidence=0.5, level="deterministic"
        )
        use_rlm, reason = _should_use_rlm_audit(extraction, source, base_report)
        assert use_rlm is False
        assert "local_model" in reason

    def test_127_0_0_1_detected_as_local(self, monkeypatch):
        from mosaicx.verify.engine import _should_use_rlm_audit

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(api_base="http://127.0.0.1:8000/v1"),
        )
        extraction = {f"field_{i}": f"value number {i} with enough length" for i in range(20)}
        source = "A" * 10000
        base_report = VerificationReport(
            verdict="partially_supported", confidence=0.5, level="deterministic"
        )
        use_rlm, reason = _should_use_rlm_audit(extraction, source, base_report)
        assert use_rlm is False
        assert "local_model" in reason

    def test_high_confidence_skips_rlm(self, monkeypatch):
        from mosaicx.verify.engine import _should_use_rlm_audit

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(api_base="https://api.openai.com/v1"),
        )
        extraction = {f"field_{i}": f"value number {i} with enough length" for i in range(20)}
        source = "A" * 10000
        base_report = VerificationReport(
            verdict="verified", confidence=0.95, level="deterministic", issues=[]
        )
        use_rlm, reason = _should_use_rlm_audit(extraction, source, base_report)
        assert use_rlm is False
        assert "high_det_confidence" in reason

    def test_cloud_model_large_doc_low_confidence_uses_rlm(self, monkeypatch):
        from mosaicx.verify.engine import _should_use_rlm_audit

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(api_base="https://api.openai.com/v1"),
        )
        extraction = {f"field_{i}": f"value number {i} with enough length" for i in range(20)}
        source = "A" * 10000
        base_report = VerificationReport(
            verdict="partially_supported", confidence=0.5, level="deterministic"
        )
        use_rlm, reason = _should_use_rlm_audit(extraction, source, base_report)
        assert use_rlm is True
        assert reason == "full_audit"

    def test_force_env_overrides_all(self, monkeypatch):
        from mosaicx.verify.engine import _should_use_rlm_audit

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(api_base="http://localhost:11434/v1"),
        )
        monkeypatch.setenv("MOSAICX_FORCE_RLM_AUDIT", "true")
        extraction = {"a": "short"}
        source = "short"
        base_report = VerificationReport(
            verdict="verified", confidence=0.99, level="deterministic"
        )
        use_rlm, reason = _should_use_rlm_audit(extraction, source, base_report)
        assert use_rlm is True
        assert reason == "forced_by_env"

    def test_claim_router_local_model_skips_rlm(self, monkeypatch):
        from mosaicx.verify.engine import _should_use_rlm_claim_audit

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(api_base="http://localhost:11434/v1"),
        )
        use_rlm, reason = _should_use_rlm_claim_audit(
            "EF was 45%", "A" * 10000
        )
        assert use_rlm is False
        assert "local_model" in reason

    def test_claim_router_short_claim_short_doc_skips_rlm(self, monkeypatch):
        from mosaicx.verify.engine import _should_use_rlm_claim_audit

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(api_base="https://api.openai.com/v1"),
        )
        use_rlm, reason = _should_use_rlm_claim_audit("EF was 45%", "Short doc.")
        assert use_rlm is False
        assert "short_claim" in reason

    def test_thorough_with_local_model_downgrades_to_spot_check(self, monkeypatch):
        """End-to-end: thorough level with local model should downgrade, not hang."""
        from mosaicx.verify.engine import _enhance_with_audit
        from mosaicx.verify.models import Issue

        monkeypatch.setattr(
            "mosaicx.config.get_config",
            lambda: MosaicxConfig(api_base="http://localhost:11434/v1"),
        )
        # Mock spot_check to avoid actual LLM call
        monkeypatch.setattr(
            "mosaicx.verify.engine._enhance_with_spot_check",
            lambda base, ext, src: VerificationReport(
                verdict="verified",
                confidence=0.85,
                level="spot_check",
                issues=[],
                field_verdicts=[],
            ),
        )
        extraction = {f"field_{i}": f"value {i} long enough text" for i in range(20)}
        source = "A" * 10000
        base_report = VerificationReport(
            verdict="partially_supported", confidence=0.5, level="deterministic"
        )
        result = _enhance_with_audit(base_report, extraction, source)
        assert result.level == "spot_check"
        assert any(i.type == "audit_downgraded" for i in result.issues)

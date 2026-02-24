"""Behavioral tests for the SDK verify() function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

CT_CHEST = Path(__file__).parent / "datasets" / "extract" / "ct_chest_sample.txt"

SAMPLE_SOURCE = (
    "CT CHEST WITH CONTRAST\n"
    "FINDINGS: 2.3 cm spiculated nodule in the right upper lobe. "
    "4 mm ground-glass nodule in the left lower lobe, stable. "
    "No pleural effusion. Heart size normal.\n"
    "IMPRESSION: 1. Growing 2.3 cm RUL nodule, suspicious. "
    "2. Stable 4 mm LLL GGN."
)

SAMPLE_EXTRACTION = {
    "findings": [
        {
            "anatomy": "right upper lobe",
            "observation": "spiculated nodule",
            "measurement": {"value": 2.3, "unit": "cm"},
            "severity": "suspicious",
        },
        {
            "anatomy": "left lower lobe",
            "observation": "ground-glass nodule",
            "measurement": {"value": 4, "unit": "mm"},
        },
    ],
    "impressions": [
        {"statement": "Growing RUL nodule", "finding_refs": [0]},
        {"statement": "Stable LLL GGN", "finding_refs": [1]},
    ],
}


class TestSDKVerifyQuick:
    """Quick SDK verify should work without any LLM."""

    def test_verify_returns_dict_with_required_keys(self):
        from mosaicx.sdk import verify

        result = verify(
            extraction={"findings": []},
            source_text="Normal report.",
            level="quick",
        )
        assert isinstance(result, dict)
        for key in (
            "verdict",
            "decision",
            "claim_truth",
            "claim_true",
            "is_verified",
            "is_contradicted",
            "confidence",
            "confidence_score",
            "level",
            "issues",
            "field_verdicts",
            "requested_level",
            "effective_level",
            "fallback_used",
            "support_score",
            "verification_mode",
            "citations",
            "sources_consulted",
        ):
            assert key in result, f"Missing key: {key}"

    def test_verify_extraction_correct_values(self):
        """Verify that correct extraction values are detected as verified."""
        from mosaicx.sdk import verify

        result = verify(
            extraction=SAMPLE_EXTRACTION,
            source_text=SAMPLE_SOURCE,
            level="quick",
        )
        assert result["verdict"] == "verified"
        assert result["level"] == "deterministic"
        assert result["confidence"] > 0.5
        assert len(result["issues"]) == 0

    def test_verify_extraction_hallucinated_value(self):
        """Hallucinated measurement should be flagged."""
        from mosaicx.sdk import verify

        bad_extraction = {
            "findings": [{"measurement": {"value": 7.5, "unit": "cm"}}]
        }
        result = verify(
            extraction=bad_extraction,
            source_text=SAMPLE_SOURCE,
            level="quick",
        )
        assert result["verdict"] == "partially_supported"
        assert any(i["type"] == "value_not_found" for i in result["issues"])

    def test_verify_claim_correct(self):
        """Correct claim should be verified."""
        from mosaicx.sdk import verify

        result = verify(
            claim="2.3 cm nodule in the right upper lobe",
            source_text=SAMPLE_SOURCE,
            level="quick",
        )
        assert result["verdict"] == "verified"
        assert result["decision"] == "verified"
        assert result["level"] == "deterministic"
        assert result["support_score"] == pytest.approx(1.0)
        assert result["verification_mode"] == "claim"
        assert result["claim_truth"] is True
        assert result["claim_true"] is True
        assert result["is_verified"] is True
        assert result["is_contradicted"] is False
        assert "claim_comparison" in result
        assert result["claim_comparison"]["claimed"]
        assert result["claim_comparison"]["grounded"] is True
        assert isinstance(result.get("citations"), list)
        assert result["citations"]
        assert result["citations"][0].get("evidence_type")

    def test_verify_claim_wrong_number(self):
        """Claim with wrong number should not be verified."""
        from mosaicx.sdk import verify

        result = verify(
            claim="5.5 cm mass in the left lung",
            source_text=SAMPLE_SOURCE,
            level="quick",
        )
        assert result["verdict"] != "verified"
        assert result["decision"] != "verified"
        assert result["claim_truth"] in (None, False)
        assert result["claim_true"] in (None, False)
        assert result["is_verified"] is False

    def test_verify_with_document_path(self):
        """Verify should accept a document path and load the file."""
        from mosaicx.sdk import verify

        if not CT_CHEST.exists():
            pytest.skip("CT chest dataset not available")

        result = verify(
            claim="2.3 cm spiculated nodule",
            document=CT_CHEST,
            level="quick",
        )
        assert result["verdict"] == "verified"
        assert result["level"] == "deterministic"


class TestSDKVerifyFallback:
    """When LLM fails, SDK verify should fall back to deterministic."""

    def test_standard_falls_back_on_llm_failure(self):
        """Standard verify should return deterministic with llm_unavailable on LLM failure."""
        from mosaicx.sdk import verify

        with patch("mosaicx.verify.spot_check.SpotChecker", side_effect=RuntimeError("LLM down")):
            result = verify(
                claim="2.3 cm nodule",
                source_text=SAMPLE_SOURCE,
                level="standard",
            )
        assert result["level"] == "deterministic"
        assert any(i["type"] == "llm_unavailable" for i in result["issues"])
        assert result["fallback_used"] is True
        assert "fallback_reason" in result

    def test_thorough_falls_back_on_llm_failure(self):
        from mosaicx.sdk import verify

        with patch("mosaicx.verify.spot_check.SpotChecker", side_effect=RuntimeError("LLM down")), \
             patch("mosaicx.verify.audit.run_audit", side_effect=RuntimeError("RLM down")):
            result = verify(
                extraction=SAMPLE_EXTRACTION,
                source_text=SAMPLE_SOURCE,
                level="thorough",
            )
        assert result["level"] == "deterministic"
        assert any(i["type"] == "llm_unavailable" for i in result["issues"])

    def test_fallback_still_detects_hallucination(self):
        """Even on LLM failure, deterministic checks should detect bad values."""
        from mosaicx.sdk import verify

        bad_extraction = {
            "findings": [{"measurement": {"value": 99, "unit": "cm"}}]
        }
        with patch("mosaicx.verify.spot_check.SpotChecker", side_effect=RuntimeError("LLM down")):
            result = verify(
                extraction=bad_extraction,
                source_text=SAMPLE_SOURCE,
                level="standard",
            )
        # Should have both value_not_found AND llm_unavailable
        issue_types = {i["type"] for i in result["issues"]}
        assert "value_not_found" in issue_types
        assert "llm_unavailable" in issue_types

    def test_claim_value_conflict_overrides_verified_decision(self):
        """Grounded claimed/source mismatch should force contradicted decision."""
        from mosaicx.sdk import verify
        from mosaicx.verify.models import VerificationReport

        mocked_report = VerificationReport(
            verdict="verified",
            confidence=0.9,
            level="deterministic",
            issues=[],
            field_verdicts=[],
        )

        with patch("mosaicx.verify.engine.verify", return_value=mocked_report):
            result = verify(
                claim="patient BP is 120/82",
                source_text="Vitals: BP 128/82 measured at triage.",
                level="quick",
            )

        assert result["decision"] == "contradicted"
        assert result["claim_true"] is False
        assert any(i["type"] == "claim_value_conflict" for i in result["issues"])

    def test_thorough_claim_matching_source_rescues_partial_verdict(self):
        """Matching grounded claim/source values should resolve to verified."""
        from mosaicx.sdk import verify
        from mosaicx.verify.models import FieldVerdict, Issue, VerificationReport

        mocked_report = VerificationReport(
            verdict="partially_supported",
            confidence=0.34,
            level="audit",
            issues=[
                Issue(
                    type="audit_inconclusive",
                    field="claim",
                    detail="The source document does not contain any blood pressure reading.",
                    severity="warning",
                )
            ],
            field_verdicts=[
                FieldVerdict(
                    status="unsupported",
                    field_path="claim",
                    claimed_value="patient BP is 128/82",
                    source_value="BP 128/82",
                    evidence_excerpt="Vitals: BP 128/82 measured at triage.",
                    evidence_source="sample_patient_vitals.pdf",
                )
            ],
        )

        with patch("mosaicx.verify.engine.verify", return_value=mocked_report):
            result = verify(
                claim="patient BP is 128/82",
                source_text="Vitals: BP 128/82 measured at triage.",
                level="thorough",
            )

        assert result["decision"] == "verified"
        assert result["claim_true"] is True
        assert result.get("match_rescued") is True
        assert result["support_score"] == pytest.approx(1.0)
        assert "128/82" in str(result["claim_comparison"]["source"])
        assert "does not contain" not in str(result["claim_comparison"]["evidence"]).lower()
        assert not any("does not contain" in str(i.get("detail", "")).lower() for i in result["issues"])
        assert result["citations"]
        assert any("128/82" in str(c.get("snippet", "")) for c in result["citations"])
        assert not any("does not contain" in str(c.get("snippet", "")).lower() for c in result["citations"])


class TestSDKVerifyEdgeCases:
    def test_verify_claim_citations_preserve_chunk_metadata(self):
        from mosaicx.sdk import verify
        from mosaicx.verify.models import FieldVerdict, VerificationReport

        mocked_report = VerificationReport(
            verdict="verified",
            confidence=0.93,
            level="audit",
            field_verdicts=[
                FieldVerdict(
                    status="verified",
                    field_path="claim",
                    claimed_value="patient BP is 128/82",
                    source_value="BP 128/82",
                    evidence_excerpt="Vitals: BP 128/82 measured at triage.",
                    evidence_source="sample_patient_vitals.pdf",
                    evidence_type="text_chunk",
                    evidence_chunk_id=9,
                    evidence_start=1220,
                    evidence_end=1268,
                    evidence_score=8.0,
                )
            ],
        )

        with patch("mosaicx.verify.engine.verify", return_value=mocked_report):
            result = verify(
                claim="patient BP is 128/82",
                source_text="Vitals: BP 128/82 measured at triage.",
                level="thorough",
            )

        assert result["citations"]
        top = result["citations"][0]
        assert top["evidence_type"] == "text_chunk"
        assert top["chunk_id"] == 9
        assert top["source"] == "sample_patient_vitals.pdf"

    def test_requires_source_text_or_document(self):
        from mosaicx.sdk import verify

        with pytest.raises(ValueError, match="source_text or document"):
            verify(claim="test")

    def test_requires_extraction_or_claim(self):
        from mosaicx.sdk import verify

        with pytest.raises(ValueError, match="extraction or claim"):
            verify(source_text="some text")

    def test_invalid_level(self):
        from mosaicx.sdk import verify

        with pytest.raises(ValueError, match="Unknown verification level"):
            verify(claim="test", source_text="text", level="mega")

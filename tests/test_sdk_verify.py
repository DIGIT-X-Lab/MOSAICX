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
        assert "claim_comparison" in result
        assert result["claim_comparison"]["claimed"]
        assert result["claim_comparison"]["grounded"] is True

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

        with patch("mosaicx.verify.spot_check.SpotChecker", side_effect=RuntimeError("LLM down")):
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


class TestSDKVerifyEdgeCases:
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

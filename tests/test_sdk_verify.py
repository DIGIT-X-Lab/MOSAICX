"""Behavioral tests for the SDK verify() function."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
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
            "result",
            "claim_is_true",
            "confidence",
            "support_score",
            "based_on_source",
            "verify_type",
            "issues",
            "field_checks",
            "requested_mode",
            "executed_mode",
            "fallback_used",
            "fallback_reason",
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
        assert result["result"] == "verified"
        assert result["executed_mode"] == "deterministic"
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
        assert result["result"] == "partially_supported"
        assert any(i["type"] == "value_not_found" for i in result["issues"])

    def test_verify_claim_correct(self):
        """Correct claim should be verified."""
        from mosaicx.sdk import verify

        result = verify(
            claim="2.3 cm nodule in the right upper lobe",
            source_text=SAMPLE_SOURCE,
            level="quick",
        )
        assert result["result"] == "verified"
        assert result["executed_mode"] == "deterministic"
        assert result["support_score"] == pytest.approx(1.0)
        assert result["verify_type"] == "claim"
        assert result["claim_is_true"] is True
        assert result["claim"]
        assert result["based_on_source"] is True
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
        assert result["result"] != "verified"
        assert result["claim_is_true"] in (None, False)

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
        assert result["result"] == "verified"
        assert result["executed_mode"] == "deterministic"


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
        assert result["executed_mode"] == "deterministic"
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
        assert result["executed_mode"] == "deterministic"
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

        assert result["result"] == "contradicted"
        assert result["claim_is_true"] is False
        assert any(i["type"] == "claim_value_conflict" for i in result["issues"])

    def test_claim_value_conflict_prefers_full_bp_pair_from_source(self):
        """BP conflicts should ground to a full BP pair, not partial numeric token."""
        from mosaicx.sdk import verify
        from mosaicx.verify.models import VerificationReport

        mocked_report = VerificationReport(
            verdict="verified",
            confidence=0.9,
            level="deterministic",
            issues=[],
            field_verdicts=[],
        )

        source_text = (
            "Measurements Vital Sign Reading Normal Range Status "
            "Blood Pressure 128/82 mmHg 120/80 mmHg Slightly Elevated."
        )

        with patch("mosaicx.verify.engine.verify", return_value=mocked_report):
            result = verify(
                claim="patient BP is 120/82",
                source_text=source_text,
                level="quick",
            )

        assert result["result"] == "contradicted"
        assert result["claim_is_true"] is False
        assert "128/82" in str(result["source_value"])
        assert "128/82" in str(result["issues"][0]["message"])

    def test_claim_grounding_recovers_when_report_echoes_claim_value(self):
        """If audit echoes claim value as source, SDK must re-ground from source text."""
        from mosaicx.sdk import verify
        from mosaicx.verify.models import FieldVerdict, VerificationReport

        mocked_report = VerificationReport(
            verdict="verified",
            confidence=0.95,
            level="audit",
            issues=[],
            field_verdicts=[
                FieldVerdict(
                    status="verified",
                    field_path="claim",
                    claimed_value="patient BP is 120/82",
                    source_value="120/82",
                    evidence_excerpt="Claim BP 120/82, source BP values found: []",
                    evidence_source="source_document",
                )
            ],
        )

        source_text = (
            "Measurements Vital Sign Reading Normal Range Status "
            "Blood Pressure 128/82 mmHg 120/80 mmHg Slightly Elevated."
        )

        with patch("mosaicx.verify.engine.verify", return_value=mocked_report):
            result = verify(
                claim="patient BP is 120/82",
                source_text=source_text,
                level="thorough",
            )

        assert result["result"] == "contradicted"
        assert result["claim_is_true"] is False
        assert result["source_value"] == "128/82"
        assert "120/82" not in str(result["evidence"])

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

        assert result["result"] == "verified"
        assert result["claim_is_true"] is True
        assert result["support_score"] == pytest.approx(1.0)
        assert "128/82" in str(result["source_value"])
        assert "does not contain" not in str(result["evidence"]).lower()
        assert not any("does not contain" in str(i.get("message", "")).lower() for i in result["issues"])
        assert result["citations"]
        assert any("128/82" in str(c.get("snippet", "")) for c in result["citations"])
        assert not any("does not contain" in str(c.get("snippet", "")).lower() for c in result["citations"])

    def test_thorough_claim_rescue_rewrites_no_information_found_evidence(self):
        """Rescued claim should not keep 'no information found' evidence text."""
        from mosaicx.sdk import verify
        from mosaicx.verify.models import FieldVerdict, Issue, VerificationReport

        mocked_report = VerificationReport(
            verdict="insufficient_evidence",
            confidence=0.35,
            level="audit",
            issues=[
                Issue(
                    type="audit_unsupported",
                    field="claim",
                    detail="No blood pressure information found in source.",
                    severity="warning",
                )
            ],
            field_verdicts=[
                FieldVerdict(
                    status="unsupported",
                    field_path="claim",
                    claimed_value="patient BP is 128/82",
                    source_value="128/82",
                    evidence_excerpt="No blood pressure information found in source.",
                    evidence_source="sample_patient_vitals.pdf",
                )
            ],
        )

        with patch("mosaicx.verify.engine.verify", return_value=mocked_report):
            result = verify(
                claim="patient BP is 128/82",
                source_text="Patient vital signs: Blood Pressure 128/82 mmHg.",
                level="thorough",
            )

        assert result["result"] == "verified"
        assert result["claim_is_true"] is True
        assert result["confidence"] >= 0.85
        assert "no blood pressure information found" not in str(result["evidence"]).lower()


class TestSDKVerifyEdgeCases:
    def test_dspy_adjudicator_prefers_multichain_output_when_available(self, monkeypatch):
        """DSPy adjudicator should accept MultiChainComparison decisions."""
        from mosaicx import sdk as sdk_mod

        class _FakeLM:
            def copy(self, **_kwargs):
                return self

        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
                return False

        class _FakeDSPY:
            settings = SimpleNamespace(lm=_FakeLM())

            @staticmethod
            def context(**_kwargs):
                return _Ctx()

            @staticmethod
            def ChainOfThought(_sig):  # noqa: ANN001
                class _Base:
                    def __call__(self, **_kwargs):
                        return SimpleNamespace(final_decision="insufficient_evidence", rationale="base")

                return _Base()

            @staticmethod
            def MultiChainComparison(_sig, M=3, temperature=0.2):  # noqa: ANN001, ARG004
                class _MCC:
                    def __call__(self, **_kwargs):
                        return SimpleNamespace(final_decision="verified")

                return _MCC()

            @staticmethod
            def BestOfN(module, N=3, reward_fn=None, threshold=0.8):  # noqa: ANN001, ARG004
                class _Best:
                    def __call__(self, **_kwargs):
                        return SimpleNamespace(final_decision="contradicted")

                return _Best()

        monkeypatch.setitem(sys.modules, "dspy", _FakeDSPY())

        decision = sdk_mod._adjudicate_claim_decision_with_dspy(
            claim="patient has severe pain",
            claim_comparison={
                "claimed": "patient has severe pain",
                "source": "pain reported in history",
                "evidence": "History: pain reported during prior visits.",
                "grounded": True,
            },
            current_decision="insufficient_evidence",
            citations=[{"source": "source_document", "snippet": "History: pain reported during prior visits."}],
        )

        assert decision == "verified"

    def test_ambiguous_grounded_claim_uses_dspy_adjudication(self):
        """Ambiguous grounded claim can be finalized by DSPy adjudication."""
        from mosaicx.sdk import verify
        from mosaicx.verify.models import FieldVerdict, VerificationReport

        mocked_report = VerificationReport(
            verdict="partially_supported",
            confidence=0.42,
            level="audit",
            issues=[],
            field_verdicts=[
                FieldVerdict(
                    status="unsupported",
                    field_path="claim",
                    claimed_value="patient has severe pain",
                    source_value="pain reported in clinical history",
                    evidence_excerpt="History: pain reported during prior visits.",
                    evidence_source="source_document",
                )
            ],
        )

        with patch("mosaicx.verify.engine.verify", return_value=mocked_report), patch(
            "mosaicx.sdk._adjudicate_claim_decision_with_dspy",
            return_value="verified",
        ) as adjudicator:
            result = verify(
                claim="patient has severe pain",
                source_text="History: pain reported during prior visits.",
                level="thorough",
            )

        adjudicator.assert_called_once()
        assert result["result"] == "verified"
        assert result["claim_is_true"] is True
        assert result["adjudication"] == "dspy_mcc_bestofn"

    def test_clear_claim_conflict_skips_dspy_adjudication(self):
        """Clear numeric conflicts must remain deterministic contradictions."""
        from mosaicx.sdk import verify
        from mosaicx.verify.models import FieldVerdict, VerificationReport

        mocked_report = VerificationReport(
            verdict="partially_supported",
            confidence=0.33,
            level="audit",
            issues=[],
            field_verdicts=[
                FieldVerdict(
                    status="mismatch",
                    field_path="claim",
                    claimed_value="patient BP is 120/82",
                    source_value="128/82",
                    evidence_excerpt="Vitals: BP 128/82 measured at triage.",
                    evidence_source="source_document",
                )
            ],
        )

        with patch("mosaicx.verify.engine.verify", return_value=mocked_report), patch(
            "mosaicx.sdk._adjudicate_claim_decision_with_dspy",
            return_value="verified",
        ) as adjudicator:
            result = verify(
                claim="patient BP is 120/82",
                source_text="Vitals: BP 128/82 measured at triage.",
                level="thorough",
            )

        adjudicator.assert_not_called()
        assert result["result"] == "contradicted"
        assert result["claim_is_true"] is False
        assert "adjudication" not in result

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

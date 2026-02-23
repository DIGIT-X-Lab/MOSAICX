"""Tests for verification data models."""
from __future__ import annotations


class TestVerificationReport:
    def test_construction(self):
        from mosaicx.verify.models import VerificationReport

        report = VerificationReport(
            verdict="verified",
            confidence=0.95,
            level="deterministic",
        )
        assert report.verdict == "verified"
        assert report.confidence == 0.95

    def test_to_dict(self):
        from mosaicx.verify.models import Issue, VerificationReport

        report = VerificationReport(
            verdict="partially_supported",
            confidence=0.6,
            level="spot_check",
            issues=[
                Issue(
                    type="value_mismatch",
                    field="findings[0].measurement",
                    detail="Claimed 22mm, source says 14mm",
                    severity="critical",
                )
            ],
        )
        d = report.to_dict()
        assert d["verdict"] == "partially_supported"
        assert len(d["issues"]) == 1
        assert d["issues"][0]["severity"] == "critical"


class TestIssue:
    def test_severity_values(self):
        from mosaicx.verify.models import Issue

        for sev in ("info", "warning", "critical"):
            issue = Issue(type="test", field="x", detail="y", severity=sev)
            assert issue.severity == sev


class TestFieldVerdict:
    def test_construction(self):
        from mosaicx.verify.models import FieldVerdict

        fv = FieldVerdict(
            status="mismatch",
            claimed_value="22mm",
            source_value="14mm",
            evidence_excerpt="nodule now measures 14mm",
            evidence_source="report.txt",
            evidence_type="text_chunk",
            evidence_chunk_id=7,
            evidence_start=420,
            evidence_end=460,
            evidence_score=8.5,
            severity="critical",
        )
        assert fv.status == "mismatch"
        assert fv.evidence_type == "text_chunk"
        assert fv.evidence_chunk_id == 7

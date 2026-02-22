# tests/test_verify_spot_check.py
from __future__ import annotations


class TestSpotCheckSignature:
    def test_verify_claim_has_expected_fields(self):
        from mosaicx.verify.spot_check import VerifyClaim

        assert "source_text" in VerifyClaim.input_fields
        assert "claims" in VerifyClaim.input_fields
        assert "verdicts" in VerifyClaim.output_fields


class TestHighRiskFieldSelector:
    def test_selects_measurements(self):
        from mosaicx.verify.spot_check import select_high_risk_fields

        extraction = {
            "exam_type": "CT Chest",
            "findings": [
                {"anatomy": "RUL", "measurement": {"value": 12, "unit": "mm"}},
                {"anatomy": "LLL", "measurement": None},
            ],
        }
        fields = select_high_risk_fields(extraction)
        assert any("measurement" in f for f in fields)

    def test_selects_severity_fields(self):
        from mosaicx.verify.spot_check import select_high_risk_fields

        extraction = {
            "findings": [{"severity": "severe", "anatomy": "C6-7"}],
        }
        fields = select_high_risk_fields(extraction)
        assert any("severity" in f for f in fields)

    def test_ignores_non_risk_fields(self):
        from mosaicx.verify.spot_check import select_high_risk_fields

        extraction = {"exam_type": "CT", "findings": [{"anatomy": "RUL", "description": "normal"}]}
        fields = select_high_risk_fields(extraction)
        assert not any("anatomy" in f for f in fields)
        assert not any("description" in f for f in fields)

    def test_empty_extraction(self):
        from mosaicx.verify.spot_check import select_high_risk_fields

        assert select_high_risk_fields({}) == []

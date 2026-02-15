"""Tests for FHIR R4 DiagnosticReport bundle."""
import pytest
import json

class TestFHIRBundle:
    def test_build_diagnostic_report(self):
        from mosaicx.schemas.fhir import build_diagnostic_report
        bundle = build_diagnostic_report(
            patient_id="P001",
            findings=[{"anatomy": "right upper lobe", "observation": "nodule"}],
            impression="Pulmonary nodule.",
            procedure_code="24627-2",
        )
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        resource_types = [e["resource"]["resourceType"] for e in bundle["entry"]]
        assert "DiagnosticReport" in resource_types
        assert "Observation" in resource_types

    def test_bundle_is_valid_json(self):
        from mosaicx.schemas.fhir import build_diagnostic_report
        bundle = build_diagnostic_report(patient_id="P001", findings=[], impression="Normal.")
        json_str = json.dumps(bundle)
        assert len(json_str) > 0

    def test_observation_per_finding(self):
        from mosaicx.schemas.fhir import build_diagnostic_report
        bundle = build_diagnostic_report(
            patient_id="P001",
            findings=[
                {"anatomy": "RUL", "observation": "nodule"},
                {"anatomy": "RLL", "observation": "atelectasis"},
            ],
            impression="Two findings.",
        )
        observations = [e for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"]
        assert len(observations) == 2

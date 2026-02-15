# tests/test_export_fhir.py
"""Tests for FHIR bundle file export."""
import pytest
import json

class TestFHIRBundleExport:
    def test_export_creates_files(self, tmp_path):
        from mosaicx.export.fhir_bundle import export_fhir_bundles
        results = [
            {"source_file": "report1.pdf", "patient_id": "P001",
             "findings": [{"anatomy": "RUL", "observation": "nodule"}],
             "impression": "Nodule."},
            {"source_file": "report2.pdf", "patient_id": "P002",
             "findings": [], "impression": "Normal."},
        ]
        export_fhir_bundles(results, tmp_path)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 2

    def test_exported_bundle_is_valid(self, tmp_path):
        from mosaicx.export.fhir_bundle import export_fhir_bundles
        results = [{"source_file": "r.pdf", "patient_id": "P001",
                     "findings": [{"anatomy": "RUL", "observation": "nodule"}],
                     "impression": "Nodule."}]
        export_fhir_bundles(results, tmp_path)
        files = list(tmp_path.glob("*.json"))
        bundle = json.loads(files[0].read_text())
        assert bundle["resourceType"] == "Bundle"

# tests/integration/test_e2e.py
"""End-to-end integration tests -- document loading -> extraction -> export.

Exercises the full v2 pipeline stack without requiring an LLM.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


@pytest.mark.integration
class TestEndToEnd:
    """Core pipeline integration tests."""

    # -- 1. Text file loading ------------------------------------------------

    def test_txt_to_json_extraction(self, tmp_path: Path) -> None:
        """Load a text file, verify document loading works."""
        from mosaicx.documents import load_document

        report = tmp_path / "report.txt"
        report.write_text(
            "Patient: John Doe, 65M.\n"
            "Indication: Persistent cough.\n"
            "Findings: 5mm ground glass nodule in the right upper lobe.\n"
            "Impression: Pulmonary nodule, recommend follow-up CT in 12 months."
        )

        doc = load_document(report)
        assert not doc.is_empty
        assert "5mm" in doc.text
        assert doc.format == "txt"
        assert doc.source_path == report

    # -- 2. Template compilation roundtrip -----------------------------------

    def test_template_compilation_roundtrip(self) -> None:
        """YAML template -> Pydantic model -> instantiation."""
        from mosaicx.schemas.template_compiler import compile_template

        yaml_content = """\
name: TestReport
description: Test
sections:
  - name: summary
    type: str
    required: true
"""
        Model = compile_template(yaml_content)
        instance = Model(summary="Normal findings.")
        assert instance.summary == "Normal findings."
        assert Model.__name__ == "TestReport"

    # -- 3. Completeness scoring ---------------------------------------------

    def test_completeness_scoring(self) -> None:
        """Completeness evaluator on a Pydantic model."""
        from pydantic import BaseModel

        from mosaicx.evaluation.completeness import compute_completeness

        class MiniReport(BaseModel):
            indication: str
            findings: list[str]

        report = MiniReport(indication="Cough", findings=["nodule"])
        result = compute_completeness(report, source_text="Cough. 5mm nodule in RUL.")

        assert 0.0 <= result["overall"] <= 1.0
        assert "field_coverage" in result
        assert "information_density" in result
        # Both fields are populated -> coverage should be 1.0
        assert result["field_coverage"] == 1.0

    # -- 4. FHIR bundle structure --------------------------------------------

    def test_fhir_bundle_structure(self) -> None:
        """FHIR bundle has valid structure."""
        from mosaicx.schemas.fhir import build_diagnostic_report

        bundle = build_diagnostic_report(
            patient_id="TEST001",
            findings=[{"anatomy": "RUL", "observation": "nodule"}],
            impression="Nodule.",
        )

        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert "entry" in bundle
        assert len(bundle["entry"]) >= 2  # 1 DiagnosticReport + 1 Observation

        # First entry should be the DiagnosticReport
        diag_report = bundle["entry"][0]["resource"]
        assert diag_report["resourceType"] == "DiagnosticReport"
        assert diag_report["conclusion"] == "Nodule."
        assert diag_report["subject"]["reference"] == "Patient/TEST001"

        # Second entry should be an Observation
        observation = bundle["entry"][1]["resource"]
        assert observation["resourceType"] == "Observation"

    # -- 5. De-identify regex roundtrip --------------------------------------

    def test_deidentify_regex_roundtrip(self, tmp_path: Path) -> None:
        """Load a text file with PHI, call regex_scrub_phi, verify PHI is removed."""
        from mosaicx.documents import load_document
        from mosaicx.pipelines.deidentifier import regex_scrub_phi

        phi_report = tmp_path / "phi_report.txt"
        phi_report.write_text(
            "Patient: Jane Smith, DOB 3/15/1960.\n"
            "SSN: 123-45-6789.\n"
            "Phone: (555) 867-5309.\n"
            "Email: jane.smith@hospital.org.\n"
            "MRN: 12345678.\n"
            "Findings: Normal chest X-ray."
        )

        doc = load_document(phi_report)
        scrubbed = regex_scrub_phi(doc.text)

        # PHI should be redacted
        assert "123-45-6789" not in scrubbed  # SSN
        assert "(555) 867-5309" not in scrubbed  # phone
        assert "jane.smith@hospital.org" not in scrubbed  # email
        assert "12345678" not in scrubbed  # MRN
        assert "3/15/1960" not in scrubbed  # date

        # Clinical content should be preserved
        assert "Normal chest X-ray" in scrubbed

        # Redaction markers should be present
        assert "[REDACTED]" in scrubbed

    # -- 6. Ontology resolver lookup -----------------------------------------

    def test_ontology_resolver_lookup(self) -> None:
        """Resolve 'right upper lobe' to RadLex code."""
        from mosaicx.schemas.ontology import OntologyResolver

        resolver = OntologyResolver()
        result = resolver.resolve("right upper lobe")

        assert result is not None
        assert result.code == "RID1303"
        assert result.vocabulary == "radlex"
        assert result.term == "right upper lobe"
        assert result.confidence == 1.0

    def test_ontology_resolver_case_insensitive(self) -> None:
        """Ontology resolver handles mixed-case input."""
        from mosaicx.schemas.ontology import OntologyResolver

        resolver = OntologyResolver()
        result = resolver.resolve("Right Upper Lobe")

        assert result is not None
        assert result.code == "RID1303"

    def test_ontology_resolver_unknown_term(self) -> None:
        """Ontology resolver returns None for unknown terms."""
        from mosaicx.schemas.ontology import OntologyResolver

        resolver = OntologyResolver()
        result = resolver.resolve("nonexistent_structure_xyz")

        assert result is None

    # -- 7. Export pipeline (CSV) --------------------------------------------

    def test_export_csv_pipeline(self, tmp_path: Path) -> None:
        """Create sample results dict, export to CSV, verify file content."""
        from mosaicx.export.tabular import export_csv

        results = [
            {
                "patient_id": "P001",
                "exam_type": "chest_ct",
                "impression": "Normal",
                "findings": [
                    {"anatomy": "RUL", "observation": "clear"},
                ],
            },
            {
                "patient_id": "P002",
                "exam_type": "brain_mri",
                "impression": "No acute findings",
                "findings": [
                    {"anatomy": "frontal lobe", "observation": "normal"},
                    {"anatomy": "temporal lobe", "observation": "normal"},
                ],
            },
        ]

        csv_path = tmp_path / "results.csv"
        export_csv(results, csv_path)

        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

        # Read back and verify content
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["patient_id"] == "P001"
        assert rows[1]["patient_id"] == "P002"
        assert "findings" in rows[0]  # findings column exists (JSON-serialised)

    def test_export_csv_findings_rows_strategy(self, tmp_path: Path) -> None:
        """CSV export with findings_rows strategy produces one row per finding."""
        from mosaicx.export.tabular import export_csv

        results = [
            {
                "patient_id": "P001",
                "findings": [
                    {"anatomy": "RUL", "observation": "nodule"},
                    {"anatomy": "RLL", "observation": "clear"},
                ],
            },
        ]

        csv_path = tmp_path / "findings.csv"
        export_csv(results, csv_path, strategy="findings_rows")

        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["anatomy"] == "RUL"
        assert rows[1]["anatomy"] == "RLL"

    # -- 8. Template registry ------------------------------------------------

    def test_template_registry_list(self) -> None:
        """List templates and verify we get a non-empty list."""
        from mosaicx.schemas.radreport.registry import list_templates

        templates = list_templates()

        assert len(templates) > 0
        names = [t.name for t in templates]
        assert "chest_ct" in names
        assert "generic" in names

    def test_template_registry_get(self) -> None:
        """Get a specific template and verify it has expected fields."""
        from mosaicx.schemas.radreport.registry import get_template

        template = get_template("chest_ct")

        assert template is not None
        assert template.name == "chest_ct"
        assert template.exam_type == "chest_ct"
        assert template.radreport_id == "RDES3"
        assert template.description != ""

    def test_template_registry_get_unknown(self) -> None:
        """Registry returns None for unknown template names."""
        from mosaicx.schemas.radreport.registry import get_template

        result = get_template("nonexistent_template_xyz")
        assert result is None


@pytest.mark.integration
class TestCrossModuleIntegration:
    """Tests that exercise multiple modules together in realistic flows."""

    def test_load_deidentify_and_score(self, tmp_path: Path) -> None:
        """Full flow: load document -> deidentify -> build model -> score."""
        from pydantic import BaseModel

        from mosaicx.documents import load_document
        from mosaicx.evaluation.completeness import compute_completeness
        from mosaicx.pipelines.deidentifier import regex_scrub_phi

        # Write a report with PHI
        report_file = tmp_path / "report.txt"
        report_file.write_text(
            "Patient: Alice Johnson, DOB 6/12/1955.\n"
            "MRN: 98765432.\n"
            "Findings: 8mm nodule in right upper lobe.\n"
            "Impression: Pulmonary nodule, follow-up recommended."
        )

        # Load and deidentify
        doc = load_document(report_file)
        clean_text = regex_scrub_phi(doc.text)

        # Verify PHI removed
        assert "98765432" not in clean_text
        assert "6/12/1955" not in clean_text

        # Build a structured model from the clinical content
        class SimpleReport(BaseModel):
            findings: str
            impression: str

        structured = SimpleReport(
            findings="8mm nodule in right upper lobe",
            impression="Pulmonary nodule, follow-up recommended",
        )

        result = compute_completeness(structured, source_text=clean_text)
        assert 0.0 <= result["overall"] <= 1.0
        assert result["field_coverage"] == 1.0

    def test_ontology_to_fhir(self) -> None:
        """Resolve anatomy term via ontology, then build a FHIR bundle."""
        from mosaicx.schemas.fhir import build_diagnostic_report
        from mosaicx.schemas.ontology import OntologyResolver

        resolver = OntologyResolver()
        term_result = resolver.resolve("pulmonary nodule")
        assert term_result is not None

        bundle = build_diagnostic_report(
            patient_id="ONT001",
            findings=[
                {
                    "anatomy": "right upper lobe",
                    "observation": term_result.term,
                    "description": f"RadLex: {term_result.code}",
                }
            ],
            impression="Pulmonary nodule identified.",
        )

        assert bundle["resourceType"] == "Bundle"
        obs = bundle["entry"][1]["resource"]
        assert term_result.term in obs["code"]["text"]
        assert term_result.code in obs["valueString"]

    def test_export_jsonl_roundtrip(self, tmp_path: Path) -> None:
        """Export to JSONL format and verify roundtrip fidelity."""
        from mosaicx.export.tabular import export_jsonl

        results = [
            {"patient_id": "P001", "findings": [{"anatomy": "lung", "observation": "clear"}]},
            {"patient_id": "P002", "findings": [{"anatomy": "heart", "observation": "normal"}]},
        ]

        jsonl_path = tmp_path / "results.jsonl"
        export_jsonl(results, jsonl_path)

        assert jsonl_path.exists()

        # Read back and verify
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["patient_id"] == "P001"
        assert parsed[1]["patient_id"] == "P002"
        assert parsed[0]["findings"][0]["anatomy"] == "lung"

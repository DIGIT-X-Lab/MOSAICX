"""Contract tests for CLI brainstorm command.

These tests define the expected behavior of the CLI brainstorm command
for interactive schema building before implementation.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from click.testing import CliRunner

from mosaicx.cli.main import cli


class TestBrainstormCommand:
    """Contract tests for the brainstorm CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_report_path(self, tmp_path: Path) -> Path:
        """Create a sample medical report for brainstorming."""
        report_path = tmp_path / "sample_report.txt"
        report_content = """
RADIOLOGY REPORT

Patient: Jane Smith
DOB: 1985-03-15
MRN: 12345678
Date: 2024-01-20
Study: CT Chest with Contrast

CLINICAL HISTORY:
45-year-old female with persistent cough and fever.

TECHNIQUE:
Axial CT images of the chest were obtained with IV contrast.

FINDINGS:
1. Bilateral lower lobe consolidations with air bronchograms
2. Small bilateral pleural effusions
3. Mediastinal lymphadenopathy, largest node measures 1.2 cm
4. No pulmonary embolism identified
5. Heart size is normal

IMPRESSION:
1. Bilateral pneumonia with associated pleural effusions
2. Reactive mediastinal lymphadenopathy
3. Recommend follow-up chest imaging in 4-6 weeks

Electronically signed by: Dr. Sarah Johnson, MD
Date: 2024-01-20 14:30:00
"""
        report_path.write_text(report_content)
        return report_path

    @pytest.fixture
    def existing_schema_path(self, tmp_path: Path) -> Path:
        """Create an existing schema file for modification."""
        schema_path = tmp_path / "existing_schema.yaml"
        schema_content = """
schema:
  findings:
    - field: "patient_name"
      type: "string"
      description: "Patient's full name"
    - field: "study_date" 
      type: "date"
      description: "Date of the study"

output:
  format: "json"
  include_confidence: false

llm:
  model: "llama2"
  temperature: 0.2
"""
        schema_path.write_text(schema_content)
        return schema_path

    def test_brainstorm_command_exists(self, runner: CliRunner) -> None:
        """Test that the brainstorm command exists and shows help."""
        result = runner.invoke(cli, ["brainstorm", "--help"])
        assert result.exit_code == 0
        assert "brainstorm" in result.output.lower()
        assert "schema" in result.output.lower()

    def test_brainstorm_basic_schema_generation(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test basic schema generation from a report."""
        output_schema_path = tmp_path / "generated_schema.yaml"
        
        # Simulate non-interactive mode by providing defaults
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--schema-output", str(output_schema_path),
            "--non-interactive"
        ])
        
        assert result.exit_code == 0
        assert output_schema_path.exists()
        
        # Verify the schema file contains expected structure
        schema_content = output_schema_path.read_text()
        assert "schema:" in schema_content
        assert "findings:" in schema_content
        assert "field:" in schema_content

    def test_brainstorm_with_existing_schema(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        existing_schema_path: Path,
        tmp_path: Path
    ) -> None:
        """Test schema enhancement from existing schema."""
        output_schema_path = tmp_path / "enhanced_schema.yaml"
        
        result = runner.invoke(cli, [
            "brainstorm", 
            "--report", str(sample_report_path),
            "--existing-schema", str(existing_schema_path),
            "--schema-output", str(output_schema_path),
            "--non-interactive"
        ])
        
        assert result.exit_code == 0
        assert output_schema_path.exists()

    def test_brainstorm_interactive_mode(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test interactive schema building mode."""
        output_schema_path = tmp_path / "interactive_schema.yaml"
        
        # Simulate interactive responses
        user_input = "\n".join([
            "y",  # Accept suggested field
            "patient_diagnosis",  # Custom field name
            "string",  # Field type
            "Primary diagnosis from the report",  # Field description
            "n",  # No more fields
            "y"   # Save schema
        ])
        
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--schema-output", str(output_schema_path)
        ], input=user_input)
        
        # Should complete successfully in interactive mode
        assert result.exit_code == 0

    def test_brainstorm_field_type_suggestions(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test that appropriate field types are suggested."""
        output_schema_path = tmp_path / "typed_schema.yaml"
        
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--schema-output", str(output_schema_path),
            "--suggest-types",
            "--non-interactive"
        ])
        
        assert result.exit_code == 0
        assert output_schema_path.exists()
        
        # Verify different field types are suggested appropriately
        schema_content = output_schema_path.read_text()
        assert "date" in schema_content or "string" in schema_content

    def test_brainstorm_custom_model(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test using custom LLM model for brainstorming."""
        output_schema_path = tmp_path / "custom_model_schema.yaml"
        
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--schema-output", str(output_schema_path),
            "--model", "llama3",
            "--non-interactive"
        ])
        
        assert result.exit_code == 0
        assert output_schema_path.exists()

    def test_brainstorm_multiple_reports(
        self,
        runner: CliRunner,
        tmp_path: Path
    ) -> None:
        """Test schema generation from multiple sample reports."""
        # Create multiple sample reports
        report1 = tmp_path / "report1.txt"
        report2 = tmp_path / "report2.txt"
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        
        report1.write_text("Patient: John Doe\nFindings: Normal chest X-ray")
        report2.write_text("Patient: Jane Smith\nFindings: Pneumonia detected")
        
        output_schema_path = tmp_path / "multi_report_schema.yaml"
        
        result = runner.invoke(cli, [
            "brainstorm",
            "--reports-dir", str(reports_dir.parent),
            "--schema-output", str(output_schema_path),
            "--non-interactive"
        ])
        
        assert result.exit_code == 0
        assert output_schema_path.exists()

    def test_brainstorm_validation_rules(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test that validation rules are suggested for fields."""
        output_schema_path = tmp_path / "validated_schema.yaml"
        
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--schema-output", str(output_schema_path),
            "--include-validation",
            "--non-interactive"
        ])
        
        assert result.exit_code == 0
        assert output_schema_path.exists()
        
        schema_content = output_schema_path.read_text()
        # Should include validation rules like required fields, patterns, etc.
        assert "required" in schema_content or "validation" in schema_content

    def test_brainstorm_confidence_calibration(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test confidence calibration for field suggestions."""
        output_schema_path = tmp_path / "calibrated_schema.yaml"
        
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--schema-output", str(output_schema_path),
            "--calibrate-confidence",
            "--non-interactive"
        ])
        
        assert result.exit_code == 0
        assert output_schema_path.exists()

    def test_brainstorm_preview_mode(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test preview mode that shows suggested schema without saving."""
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--preview-only"
        ])
        
        assert result.exit_code == 0
        # Should show schema in console output
        assert "schema:" in result.output or "field:" in result.output

    def test_brainstorm_missing_report_error(self, runner: CliRunner) -> None:
        """Test error handling for missing report file."""
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", "nonexistent_report.txt",
            "--schema-output", "schema.yaml"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_brainstorm_invalid_existing_schema(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test error handling for invalid existing schema."""
        invalid_schema = tmp_path / "invalid_schema.yaml"
        invalid_schema.write_text("invalid: yaml: content: [")
        
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--existing-schema", str(invalid_schema),
            "--schema-output", "output.yaml"
        ])
        
        assert result.exit_code != 0

    def test_brainstorm_output_formats(
        self,
        runner: CliRunner,
        sample_report_path: Path,
        tmp_path: Path
    ) -> None:
        """Test different output formats for schema files."""
        # Test YAML output (default)
        yaml_output = tmp_path / "schema.yaml"
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--schema-output", str(yaml_output),
            "--format", "yaml",
            "--non-interactive"
        ])
        assert result.exit_code == 0
        assert yaml_output.exists()

        # Test JSON output
        json_output = tmp_path / "schema.json"
        result = runner.invoke(cli, [
            "brainstorm",
            "--report", str(sample_report_path),
            "--schema-output", str(json_output),
            "--format", "json",
            "--non-interactive"
        ])
        assert result.exit_code == 0
        assert json_output.exists()

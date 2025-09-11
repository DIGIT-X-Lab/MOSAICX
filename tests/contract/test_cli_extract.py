"""Contract tests for CLI extract command.

These tests define the expected behavior of the CLI extract command
before implementation. They serve as the contract that the implementation
must fulfill.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from click.testing import CliRunner

from mosaicx.cli.main import cli


class TestExtractCommand:
    """Contract tests for the extract CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_pdf_path(self, tmp_path: Path) -> Path:
        """Create a sample PDF file path."""
        pdf_path = tmp_path / "sample_report.pdf"
        # Create a dummy PDF file for testing
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n%%EOF")
        return pdf_path

    @pytest.fixture
    def sample_text_path(self, tmp_path: Path) -> Path:
        """Create a sample text file."""
        text_path = tmp_path / "sample_report.txt"
        text_path.write_text(
            "Patient: John Doe\n"
            "Date: 2024-01-15\n"
            "Findings: Bilateral lung infiltrates consistent with pneumonia.\n"
            "Impression: Acute pneumonia, recommend antibiotic treatment.\n"
        )
        return text_path

    @pytest.fixture
    def sample_config_path(self, tmp_path: Path) -> Path:
        """Create a sample configuration file."""
        config_path = tmp_path / "config.yaml"
        config_content = """
schema:
  findings:
    - field: "patient_name"
      type: "string"
      description: "Patient's full name"
    - field: "diagnosis"
      type: "string" 
      description: "Primary diagnosis"
    - field: "date"
      type: "date"
      description: "Report date"

output:
  format: "json"
  include_confidence: true
  include_source_text: true

llm:
  model: "llama2"
  temperature: 0.1
  max_tokens: 1000
"""
        config_path.write_text(config_content)
        return config_path

    def test_extract_command_exists(self, runner: CliRunner) -> None:
        """Test that the extract command exists and shows help."""
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "extract" in result.output.lower()

    def test_extract_pdf_with_config_success(
        self, 
        runner: CliRunner, 
        sample_pdf_path: Path, 
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test successful PDF extraction with configuration."""
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(cli, [
            "extract",
            str(sample_pdf_path),
            "--config", str(sample_config_path),
            "--output", str(output_path)
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        # Verify output is valid JSON
        output_data = json.loads(output_path.read_text())
        assert isinstance(output_data, dict)
        assert "extraction_results" in output_data
        assert "metadata" in output_data

    def test_extract_text_file_success(
        self,
        runner: CliRunner,
        sample_text_path: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test successful text file extraction."""
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(cli, [
            "extract",
            str(sample_text_path),
            "--config", str(sample_config_path),
            "--output", str(output_path)
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()

    def test_extract_missing_file_error(
        self, 
        runner: CliRunner, 
        sample_config_path: Path
    ) -> None:
        """Test error handling for missing input file."""
        result = runner.invoke(cli, [
            "extract",
            "nonexistent_file.pdf",
            "--config", str(sample_config_path),
            "--output", "output.json"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_extract_missing_config_error(
        self, 
        runner: CliRunner, 
        sample_pdf_path: Path
    ) -> None:
        """Test error handling for missing configuration file."""
        result = runner.invoke(cli, [
            "extract",
            str(sample_pdf_path),
            "--config", "nonexistent_config.yaml",
            "--output", "output.json"
        ])
        
        assert result.exit_code != 0
        assert "config" in result.output.lower() or "error" in result.output.lower()

    def test_extract_invalid_config_error(
        self, 
        runner: CliRunner, 
        sample_pdf_path: Path,
        tmp_path: Path
    ) -> None:
        """Test error handling for invalid configuration file."""
        invalid_config_path = tmp_path / "invalid_config.yaml"
        invalid_config_path.write_text("invalid: yaml: content: [")
        
        result = runner.invoke(cli, [
            "extract",
            str(sample_pdf_path),
            "--config", str(invalid_config_path),
            "--output", "output.json"
        ])
        
        assert result.exit_code != 0

    def test_extract_output_formats(
        self,
        runner: CliRunner,
        sample_text_path: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test different output formats (JSON, CSV)."""
        # Test JSON output
        json_output = tmp_path / "output.json"
        result = runner.invoke(cli, [
            "extract",
            str(sample_text_path),
            "--config", str(sample_config_path),
            "--output", str(json_output),
            "--format", "json"
        ])
        assert result.exit_code == 0
        assert json_output.exists()

        # Test CSV output
        csv_output = tmp_path / "output.csv"
        result = runner.invoke(cli, [
            "extract",
            str(sample_text_path),
            "--config", str(sample_config_path),
            "--output", str(csv_output),
            "--format", "csv"
        ])
        assert result.exit_code == 0
        assert csv_output.exists()

    def test_extract_verbose_output(
        self,
        runner: CliRunner,
        sample_text_path: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test verbose flag provides detailed output."""
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(cli, [
            "extract",
            str(sample_text_path),
            "--config", str(sample_config_path),
            "--output", str(output_path),
            "--verbose"
        ])
        
        assert result.exit_code == 0
        # Verbose should provide more detailed console output
        assert len(result.output) > 0

    def test_extract_dry_run(
        self,
        runner: CliRunner,
        sample_text_path: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test dry run flag doesn't create output file."""
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(cli, [
            "extract",
            str(sample_text_path),
            "--config", str(sample_config_path),
            "--output", str(output_path),
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        assert not output_path.exists()
        assert "dry run" in result.output.lower() or "preview" in result.output.lower()

    def test_extract_confidence_threshold(
        self,
        runner: CliRunner,
        sample_text_path: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test confidence threshold filtering."""
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(cli, [
            "extract",
            str(sample_text_path),
            "--config", str(sample_config_path),
            "--output", str(output_path),
            "--confidence-threshold", "0.8"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()

    def test_extract_model_override(
        self,
        runner: CliRunner,
        sample_text_path: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test overriding LLM model from command line."""
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(cli, [
            "extract",
            str(sample_text_path),
            "--config", str(sample_config_path),
            "--output", str(output_path),
            "--model", "llama3"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()

    def test_extract_progress_bar(
        self,
        runner: CliRunner,
        sample_text_path: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test that progress information is shown during extraction."""
        output_path = tmp_path / "output.json"
        
        result = runner.invoke(cli, [
            "extract",
            str(sample_text_path),
            "--config", str(sample_config_path),
            "--output", str(output_path),
            "--progress"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()

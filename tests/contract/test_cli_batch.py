"""Contract tests for CLI extract-batch command.

These tests define the expected behavior of the CLI extract-batch command
for batch processing multiple reports before implementation.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from mosaicx.cli.main import cli


class TestExtractBatchCommand:
    """Contract tests for the extract-batch CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_reports_dir(self, tmp_path: Path) -> Path:
        """Create a directory with sample medical reports."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        
        # Create multiple sample reports
        reports_data = [
            {
                "filename": "report_001.txt",
                "content": """
Patient: Alice Johnson
Date: 2024-01-15
Study: Chest X-ray
Findings: Clear lungs, no acute findings
Impression: Normal chest radiograph
"""
            },
            {
                "filename": "report_002.pdf",
                "content": b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n%%EOF"
            },
            {
                "filename": "report_003.txt", 
                "content": """
Patient: Bob Wilson
Date: 2024-01-16
Study: CT Abdomen
Findings: Hepatomegaly, no masses detected
Impression: Enlarged liver, recommend follow-up
"""
            },
            {
                "filename": "report_004.txt",
                "content": """
Patient: Carol Davis
Date: 2024-01-17
Study: MRI Brain
Findings: Small white matter lesions
Impression: Age-related changes, no acute abnormalities
"""
            }
        ]
        
        for report_data in reports_data:
            report_path = reports_dir / report_data["filename"]
            if isinstance(report_data["content"], str):
                report_path.write_text(report_data["content"])
            else:
                report_path.write_bytes(report_data["content"])
        
        return reports_dir

    @pytest.fixture
    def sample_config_path(self, tmp_path: Path) -> Path:
        """Create a sample configuration file."""
        config_path = tmp_path / "batch_config.yaml"
        config_content = """
schema:
  findings:
    - field: "patient_name"
      type: "string"
      description: "Patient's full name"
    - field: "study_type"
      type: "string"
      description: "Type of medical study"
    - field: "findings"
      type: "string"
      description: "Key findings from the report"
    - field: "impression"
      type: "string"
      description: "Clinical impression"
    - field: "date"
      type: "date"
      description: "Study date"

output:
  format: "json"
  include_confidence: true
  include_source_text: false

llm:
  model: "llama2"
  temperature: 0.1
  max_tokens: 1500

batch:
  max_workers: 4
  timeout_per_report: 300
  skip_errors: true
"""
        config_path.write_text(config_content)
        return config_path

    def test_extract_batch_command_exists(self, runner: CliRunner) -> None:
        """Test that the extract-batch command exists and shows help."""
        result = runner.invoke(cli, ["extract-batch", "--help"])
        assert result.exit_code == 0
        assert "extract-batch" in result.output.lower()
        assert "batch" in result.output.lower() or "multiple" in result.output.lower()

    def test_extract_batch_basic_success(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test basic batch processing of multiple reports."""
        output_dir = tmp_path / "batch_output"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert output_dir.exists()
        
        # Check that output files were created
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) > 0

    def test_extract_batch_with_pattern_filter(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test batch processing with file pattern filtering."""
        output_dir = tmp_path / "filtered_output"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--pattern", "*.txt"
        ])
        
        assert result.exit_code == 0
        # Should only process .txt files
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) >= 2  # At least the .txt files

    def test_extract_batch_parallel_processing(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test parallel processing with specified worker count."""
        output_dir = tmp_path / "parallel_output"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--max-workers", "2"
        ])
        
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_extract_batch_progress_reporting(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test progress reporting during batch processing."""
        output_dir = tmp_path / "progress_output"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--progress"
        ])
        
        assert result.exit_code == 0
        # Should show progress information in output
        assert "progress" in result.output.lower() or "processing" in result.output.lower()

    def test_extract_batch_summary_report(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test generation of batch processing summary report."""
        output_dir = tmp_path / "summary_output"
        output_dir.mkdir()
        summary_path = tmp_path / "batch_summary.json"
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--summary", str(summary_path)
        ])
        
        assert result.exit_code == 0
        assert summary_path.exists()
        
        # Verify summary contains expected information
        summary_data = json.loads(summary_path.read_text())
        assert "total_processed" in summary_data
        assert "successful" in summary_data
        assert "failed" in summary_data
        assert "processing_time" in summary_data

    def test_extract_batch_error_handling(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_config_path: Path
    ) -> None:
        """Test error handling with corrupted files."""
        reports_dir = tmp_path / "reports_with_errors"
        reports_dir.mkdir()
        output_dir = tmp_path / "error_output"
        output_dir.mkdir()
        
        # Create a mix of valid and invalid files
        (reports_dir / "valid_report.txt").write_text("Patient: Test\nFindings: Normal")
        (reports_dir / "corrupted_file.pdf").write_bytes(b"Invalid PDF content")
        (reports_dir / "empty_file.txt").write_text("")
        
        result = runner.invoke(cli, [
            "extract-batch", 
            str(reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--skip-errors"
        ])
        
        assert result.exit_code == 0
        # Should continue processing despite errors
        assert "error" in result.output.lower() or "skipped" in result.output.lower()

    def test_extract_batch_resume_functionality(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test resume functionality for interrupted batch processing."""
        output_dir = tmp_path / "resume_output"
        output_dir.mkdir()
        
        # Create a checkpoint file to simulate partial processing
        checkpoint_path = tmp_path / "batch_checkpoint.json"
        checkpoint_data = {
            "processed_files": ["report_001.txt"],
            "total_files": 4,
            "start_time": "2024-01-20T10:00:00Z"
        }
        checkpoint_path.write_text(json.dumps(checkpoint_data))
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--resume", str(checkpoint_path)
        ])
        
        assert result.exit_code == 0

    def test_extract_batch_output_formats(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test different output formats for batch processing."""
        # Test JSON output (default)
        json_output_dir = tmp_path / "json_output"
        json_output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(json_output_dir),
            "--format", "json"
        ])
        assert result.exit_code == 0
        assert len(list(json_output_dir.glob("*.json"))) > 0

        # Test CSV output
        csv_output_dir = tmp_path / "csv_output"
        csv_output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(csv_output_dir),
            "--format", "csv"
        ])
        assert result.exit_code == 0
        assert len(list(csv_output_dir.glob("*.csv"))) > 0

    def test_extract_batch_aggregated_output(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test aggregated output combining all results into single file."""
        aggregated_output = tmp_path / "aggregated_results.json"
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--aggregated-output", str(aggregated_output)
        ])
        
        assert result.exit_code == 0
        assert aggregated_output.exists()
        
        # Verify aggregated structure
        aggregated_data = json.loads(aggregated_output.read_text())
        assert "batch_results" in aggregated_data
        assert isinstance(aggregated_data["batch_results"], list)

    def test_extract_batch_patient_grouping(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test grouping results by patient for multi-report analysis."""
        output_dir = tmp_path / "patient_grouped_output"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--group-by-patient"
        ])
        
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_extract_batch_confidence_filtering(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test filtering results by confidence threshold."""
        output_dir = tmp_path / "confidence_filtered_output"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--confidence-threshold", "0.75"
        ])
        
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_extract_batch_missing_input_dir(self, runner: CliRunner) -> None:
        """Test error handling for missing input directory."""
        result = runner.invoke(cli, [
            "extract-batch",
            "nonexistent_directory",
            "--config", "config.yaml",
            "--output-dir", "output"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "directory" in result.output.lower()

    def test_extract_batch_missing_config(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        tmp_path: Path
    ) -> None:
        """Test error handling for missing configuration file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", "nonexistent_config.yaml",
            "--output-dir", str(output_dir)
        ])
        
        assert result.exit_code != 0
        assert "config" in result.output.lower() or "not found" in result.output.lower()

    def test_extract_batch_dry_run(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test dry run mode for batch processing."""
        output_dir = tmp_path / "dry_run_output"
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        # Output directory should not be created in dry run
        assert not output_dir.exists()
        assert "dry run" in result.output.lower() or "preview" in result.output.lower()

    def test_extract_batch_timeout_handling(
        self,
        runner: CliRunner,
        sample_reports_dir: Path,
        sample_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test timeout handling for slow processing."""
        output_dir = tmp_path / "timeout_output"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "extract-batch",
            str(sample_reports_dir),
            "--config", str(sample_config_path),
            "--output-dir", str(output_dir),
            "--timeout", "1"  # Very short timeout
        ])
        
        # Should handle timeouts gracefully
        assert result.exit_code == 0 or "timeout" in result.output.lower()

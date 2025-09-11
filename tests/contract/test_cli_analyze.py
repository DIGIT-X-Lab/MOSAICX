"""Contract tests for CLI analyze command.

These tests define the expected behavior of the CLI analyze command
for multi-report patient history analysis before implementation.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from mosaicx.cli.main import cli


class TestAnalyzeCommand:
    """Contract tests for the analyze CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click CLI runner."""
        return CliRunner()

    @pytest.fixture
    def patient_reports_dir(self, tmp_path: Path) -> Path:
        """Create a directory with multiple reports for the same patient."""
        reports_dir = tmp_path / "patient_reports"
        reports_dir.mkdir()
        
        # Create temporal sequence of reports for same patient
        reports_data = [
            {
                "filename": "john_doe_2024_01_15.txt",
                "content": """
Patient: John Doe
DOB: 1980-05-20
MRN: 12345
Date: 2024-01-15
Study: Chest X-ray
Findings: Small nodule in right upper lobe, 8mm
Impression: Pulmonary nodule, recommend follow-up CT
"""
            },
            {
                "filename": "john_doe_2024_02_20.txt", 
                "content": """
Patient: John Doe
DOB: 1980-05-20
MRN: 12345
Date: 2024-02-20
Study: CT Chest
Findings: Right upper lobe nodule now measures 12mm, increased from prior
Impression: Growing pulmonary nodule, concerning for malignancy
Recommendation: PET scan and pulmonology referral
"""
            },
            {
                "filename": "john_doe_2024_03_10.txt",
                "content": """
Patient: John Doe
DOB: 1980-05-20  
MRN: 12345
Date: 2024-03-10
Study: PET-CT
Findings: Hypermetabolic nodule in right upper lobe, SUVmax 4.2
Additional: No distant metastases identified
Impression: PET-positive pulmonary nodule, highly suspicious for primary lung cancer
"""
            },
            {
                "filename": "john_doe_2024_03_25.txt",
                "content": """
Patient: John Doe
DOB: 1980-05-20
MRN: 12345
Date: 2024-03-25
Study: Post-surgical CT Chest
Findings: Status post right upper lobe wedge resection
Post-surgical changes with no residual nodule
Impression: Successful resection of pulmonary nodule
Pathology: Adenocarcinoma, Stage IA
"""
            }
        ]
        
        for report_data in reports_data:
            report_path = reports_dir / report_data["filename"]
            report_path.write_text(report_data["content"])
        
        return reports_dir

    @pytest.fixture
    def multi_patient_reports_dir(self, tmp_path: Path) -> Path:
        """Create reports for multiple patients to test patient grouping."""
        reports_dir = tmp_path / "multi_patient_reports"
        reports_dir.mkdir()
        
        # Patient A reports
        (reports_dir / "alice_smith_001.txt").write_text("""
Patient: Alice Smith
MRN: 11111
Date: 2024-01-10
Study: Mammography
Findings: BIRADS 2, benign findings
""")
        
        (reports_dir / "alice_smith_002.txt").write_text("""
Patient: Alice Smith  
MRN: 11111
Date: 2024-07-15
Study: Mammography
Findings: BIRADS 1, negative
""")
        
        # Patient B reports
        (reports_dir / "bob_jones_001.txt").write_text("""
Patient: Bob Jones
MRN: 22222
Date: 2024-02-05
Study: Colonoscopy
Findings: Multiple polyps removed
""")
        
        return reports_dir

    @pytest.fixture
    def analysis_config_path(self, tmp_path: Path) -> Path:
        """Create configuration for patient analysis."""
        config_path = tmp_path / "analysis_config.yaml"
        config_content = """
analysis:
  type: "patient_timeline"
  patient_identification:
    - "patient_name"
    - "mrn"
    - "date_of_birth"
  
  timeline_analysis:
    sort_by: "date"
    detect_progression: true
    identify_patterns: true
    flag_concerning_changes: true
    
  synthesis:
    generate_summary: true
    highlight_key_events: true
    suggest_follow_up: true
    
schema:
  findings:
    - field: "patient_name"
      type: "string"
      description: "Patient's full name"
    - field: "mrn"
      type: "string" 
      description: "Medical record number"
    - field: "date"
      type: "date"
      description: "Study date"
    - field: "study_type"
      type: "string"
      description: "Type of study performed"
    - field: "key_findings"
      type: "string"
      description: "Important findings"
    - field: "impression"
      type: "string"
      description: "Clinical impression"

output:
  format: "json"
  include_timeline: true
  include_synthesis: true
  include_confidence: true

llm:
  model: "llama3"
  temperature: 0.2
  max_tokens: 2000
"""
        config_path.write_text(config_content)
        return config_path

    def test_analyze_command_exists(self, runner: CliRunner) -> None:
        """Test that the analyze command exists and shows help."""
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output.lower()
        assert "patient" in result.output.lower() or "timeline" in result.output.lower()

    def test_analyze_patient_timeline(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test basic patient timeline analysis."""
        output_path = tmp_path / "patient_timeline.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--patient-id", "12345"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        # Verify output structure
        analysis_data = json.loads(output_path.read_text())
        assert "patient_timeline" in analysis_data
        assert "synthesis" in analysis_data
        assert "key_events" in analysis_data

    def test_analyze_progression_detection(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test detection of disease progression."""
        output_path = tmp_path / "progression_analysis.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--detect-progression"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        analysis_data = json.loads(output_path.read_text())
        assert "progression_analysis" in analysis_data
        # Should detect nodule growth progression
        assert "changes_detected" in analysis_data["progression_analysis"]

    def test_analyze_multi_patient_grouping(
        self,
        runner: CliRunner,
        multi_patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test automatic patient grouping and separate analysis."""
        output_dir = tmp_path / "multi_patient_analysis"
        output_dir.mkdir()
        
        result = runner.invoke(cli, [
            "analyze",
            str(multi_patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output-dir", str(output_dir),
            "--group-by-patient"
        ])
        
        assert result.exit_code == 0
        
        # Should create separate analysis files for each patient
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) >= 2  # At least two patients

    def test_analyze_summary_generation(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test generation of patient history summary."""
        output_path = tmp_path / "patient_summary.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--generate-summary"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        analysis_data = json.loads(output_path.read_text())
        assert "patient_summary" in analysis_data
        assert "key_medical_events" in analysis_data["patient_summary"]

    def test_analyze_pattern_recognition(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test medical pattern recognition across reports."""
        output_path = tmp_path / "pattern_analysis.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--identify-patterns"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        analysis_data = json.loads(output_path.read_text())
        assert "patterns_identified" in analysis_data

    def test_analyze_risk_assessment(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test risk assessment based on timeline analysis."""
        output_path = tmp_path / "risk_assessment.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--assess-risk"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        analysis_data = json.loads(output_path.read_text())
        assert "risk_assessment" in analysis_data

    def test_analyze_follow_up_recommendations(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test generation of follow-up recommendations."""
        output_path = tmp_path / "followup_recs.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--suggest-followup"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        analysis_data = json.loads(output_path.read_text())
        assert "follow_up_recommendations" in analysis_data

    def test_analyze_comparative_analysis(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test comparative analysis between time points."""
        output_path = tmp_path / "comparative_analysis.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--compare-timepoints"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        analysis_data = json.loads(output_path.read_text())
        assert "comparative_analysis" in analysis_data

    def test_analyze_date_range_filtering(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test filtering analysis by date range."""
        output_path = tmp_path / "date_filtered_analysis.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--date-from", "2024-02-01",
            "--date-to", "2024-03-31"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()

    def test_analyze_confidence_weighting(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test confidence-weighted analysis."""
        output_path = tmp_path / "confidence_weighted.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--weight-by-confidence"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()

    def test_analyze_visualization_data(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test generation of data for timeline visualization."""
        output_path = tmp_path / "viz_data.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--include-viz-data"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        analysis_data = json.loads(output_path.read_text())
        assert "visualization_data" in analysis_data

    def test_analyze_missing_reports_directory(self, runner: CliRunner) -> None:
        """Test error handling for missing reports directory."""
        result = runner.invoke(cli, [
            "analyze",
            "nonexistent_directory",
            "--config", "config.yaml",
            "--output", "analysis.json"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "directory" in result.output.lower()

    def test_analyze_invalid_patient_id(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test handling of invalid patient ID."""
        output_path = tmp_path / "invalid_patient.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--patient-id", "99999"  # Non-existent patient
        ])
        
        assert result.exit_code != 0
        assert "patient" in result.output.lower() and "not found" in result.output.lower()

    def test_analyze_output_formats(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test different output formats for analysis results."""
        # Test JSON output
        json_output = tmp_path / "analysis.json"
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(json_output),
            "--format", "json"
        ])
        assert result.exit_code == 0
        assert json_output.exists()

        # Test HTML report output
        html_output = tmp_path / "analysis.html"
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(html_output),
            "--format", "html"
        ])
        assert result.exit_code == 0
        assert html_output.exists()

    def test_analyze_dry_run(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test dry run mode for analysis."""
        output_path = tmp_path / "dry_run_analysis.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        assert not output_path.exists()
        assert "dry run" in result.output.lower() or "preview" in result.output.lower()

    def test_analyze_verbose_output(
        self,
        runner: CliRunner,
        patient_reports_dir: Path,
        analysis_config_path: Path,
        tmp_path: Path
    ) -> None:
        """Test verbose analysis output."""
        output_path = tmp_path / "verbose_analysis.json"
        
        result = runner.invoke(cli, [
            "analyze",
            str(patient_reports_dir),
            "--config", str(analysis_config_path),
            "--output", str(output_path),
            "--verbose"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        assert len(result.output) > 0  # Should provide detailed console output

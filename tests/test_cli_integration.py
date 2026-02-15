# tests/test_cli_integration.py
"""Integration tests for wired CLI commands.

Tests commands that do NOT require an LLM:
    - config show
    - template list
    - template validate (valid + invalid YAML)
    - deidentify --regex-only

Tests that LLM-dependent commands fail gracefully when no API key is set:
    - extract (no API key)
    - summarize (no API key)
    - deidentify without --regex-only (no API key)
    - schema generate (no API key)
    - optimize (displays config without crashing)
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from mosaicx.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tmp_text_file(tmp_path: Path) -> Path:
    """Create a temp .txt file with sample clinical content."""
    p = tmp_path / "sample_report.txt"
    p.write_text(
        "Patient: John Doe, MRN: 12345678, DOB: 1/2/1980.\n"
        "Phone: (555) 123-4567. SSN: 123-45-6789.\n"
        "Email: john.doe@hospital.com\n"
        "Findings: 2cm nodule in the right upper lobe.\n"
        "Impression: Suspicious pulmonary nodule.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def tmp_valid_template(tmp_path: Path) -> Path:
    """Create a valid YAML template file."""
    p = tmp_path / "valid_template.yaml"
    p.write_text(
        textwrap.dedent(
            """\
            name: TestReport
            description: A test template
            sections:
              - name: indication
                type: str
                required: true
                description: Clinical indication
              - name: findings
                type: list
                item:
                  type: str
                description: List of findings
              - name: impression
                type: str
                required: true
                description: Overall impression
            """
        ),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def tmp_invalid_template(tmp_path: Path) -> Path:
    """Create an invalid YAML template file."""
    p = tmp_path / "invalid_template.yaml"
    p.write_text("this is not: [valid: yaml template", encoding="utf-8")
    return p


@pytest.fixture
def _no_api_key(monkeypatch):
    """Ensure no API key is set for DSPy-dependent tests."""
    # Set to empty string (not delete) so env var overrides any .env file value
    monkeypatch.setenv("MOSAICX_API_KEY", "")
    # Clear the lru_cache so config is re-created without any stale key
    from mosaicx.config import get_config

    get_config.cache_clear()
    yield
    get_config.cache_clear()


# -------------------------------------------------------------------------
# config show
# -------------------------------------------------------------------------


class TestConfigShow:
    def test_config_show_succeeds(self, runner: CliRunner, _no_api_key):
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "Medical Computational Suite for" in result.output

    def test_config_show_displays_fields(self, runner: CliRunner, _no_api_key):
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        # Check key config fields appear
        assert "lm" in result.output
        assert "batch_workers" in result.output
        assert "home_dir" in result.output

    def test_config_show_displays_derived_paths(self, runner: CliRunner, _no_api_key):
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "schema_dir" in result.output
        assert "optimized_dir" in result.output
        assert "checkpoint_dir" in result.output
        assert "log_dir" in result.output


# -------------------------------------------------------------------------
# template list
# -------------------------------------------------------------------------


class TestTemplateList:
    def test_template_list_succeeds(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "list"])
        assert result.exit_code == 0
        assert "TEMPLATES" in result.output

    def test_template_list_shows_builtin_templates(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "list"])
        assert result.exit_code == 0
        # Verify some known built-in template names appear
        assert "generic" in result.output
        assert "chest_ct" in result.output
        assert "brain_mri" in result.output

    def test_template_list_shows_columns(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "list"])
        assert result.exit_code == 0
        # Table headers
        assert "Name" in result.output
        assert "Exam Type" in result.output
        assert "Description" in result.output


# -------------------------------------------------------------------------
# template validate
# -------------------------------------------------------------------------


class TestTemplateValidate:
    def test_validate_valid_template(
        self, runner: CliRunner, tmp_valid_template: Path
    ):
        result = runner.invoke(
            cli, ["template", "validate", "--file", str(tmp_valid_template)]
        )
        assert result.exit_code == 0
        assert "Template is valid" in result.output
        assert "TestReport" in result.output

    def test_validate_shows_fields(
        self, runner: CliRunner, tmp_valid_template: Path
    ):
        result = runner.invoke(
            cli, ["template", "validate", "--file", str(tmp_valid_template)]
        )
        assert result.exit_code == 0
        assert "indication" in result.output
        assert "findings" in result.output
        assert "impression" in result.output

    def test_validate_invalid_template(
        self, runner: CliRunner, tmp_invalid_template: Path
    ):
        result = runner.invoke(
            cli, ["template", "validate", "--file", str(tmp_invalid_template)]
        )
        assert result.exit_code != 0
        assert "Template validation failed" in result.output

    def test_validate_missing_file(self, runner: CliRunner, tmp_path: Path):
        missing = tmp_path / "nonexistent.yaml"
        result = runner.invoke(
            cli, ["template", "validate", "--file", str(missing)]
        )
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_validate_requires_file_option(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "validate"])
        assert result.exit_code != 0
        # Click should complain about missing required option


# -------------------------------------------------------------------------
# deidentify --regex-only
# -------------------------------------------------------------------------


class TestDeidentifyRegexOnly:
    def test_regex_only_removes_ssn(
        self, runner: CliRunner, tmp_text_file: Path
    ):
        result = runner.invoke(
            cli,
            ["deidentify", "--document", str(tmp_text_file), "--regex-only"],
        )
        assert result.exit_code == 0
        assert "[REDACTED]" in result.output
        # SSN should be scrubbed
        assert "123-45-6789" not in result.output

    def test_regex_only_removes_phone(
        self, runner: CliRunner, tmp_text_file: Path
    ):
        result = runner.invoke(
            cli,
            ["deidentify", "--document", str(tmp_text_file), "--regex-only"],
        )
        assert result.exit_code == 0
        assert "(555) 123-4567" not in result.output

    def test_regex_only_removes_email(
        self, runner: CliRunner, tmp_text_file: Path
    ):
        result = runner.invoke(
            cli,
            ["deidentify", "--document", str(tmp_text_file), "--regex-only"],
        )
        assert result.exit_code == 0
        assert "john.doe@hospital.com" not in result.output

    def test_regex_only_removes_mrn(
        self, runner: CliRunner, tmp_text_file: Path
    ):
        result = runner.invoke(
            cli,
            ["deidentify", "--document", str(tmp_text_file), "--regex-only"],
        )
        assert result.exit_code == 0
        assert "MRN: 12345678" not in result.output

    def test_regex_only_preserves_clinical_content(
        self, runner: CliRunner, tmp_text_file: Path
    ):
        result = runner.invoke(
            cli,
            ["deidentify", "--document", str(tmp_text_file), "--regex-only"],
        )
        assert result.exit_code == 0
        assert "nodule" in result.output
        assert "right upper lobe" in result.output

    def test_regex_only_shows_filename_header(
        self, runner: CliRunner, tmp_text_file: Path
    ):
        result = runner.invoke(
            cli,
            ["deidentify", "--document", str(tmp_text_file), "--regex-only"],
        )
        assert result.exit_code == 0
        assert "sample_report.txt" in result.output

    def test_regex_only_directory(self, runner: CliRunner, tmp_path: Path):
        """Regex-only mode works with a directory of files."""
        for i in range(3):
            p = tmp_path / f"report_{i}.txt"
            p.write_text(f"Patient phone: (555) 000-{i:04d}\nFindings: normal.\n")

        result = runner.invoke(
            cli,
            ["deidentify", "--dir", str(tmp_path), "--regex-only"],
        )
        assert result.exit_code == 0
        # All three filenames should appear
        assert "report_0.txt" in result.output
        assert "report_1.txt" in result.output
        assert "report_2.txt" in result.output
        # Phone numbers scrubbed
        assert "(555) 000-" not in result.output

    def test_deidentify_no_input_shows_error(self, runner: CliRunner):
        result = runner.invoke(cli, ["deidentify", "--regex-only"])
        assert result.exit_code != 0
        assert "Provide --document or --dir" in result.output

    def test_deidentify_missing_document(self, runner: CliRunner, tmp_path: Path):
        missing = tmp_path / "nonexistent.txt"
        result = runner.invoke(
            cli,
            ["deidentify", "--document", str(missing), "--regex-only"],
        )
        assert result.exit_code != 0
        assert "Document not found" in result.output


# -------------------------------------------------------------------------
# optimize (no LLM needed for config display)
# -------------------------------------------------------------------------


class TestOptimize:
    def test_optimize_shows_config(self, runner: CliRunner):
        result = runner.invoke(cli, ["optimize", "--budget", "light"])
        assert result.exit_code == 0
        assert "OPTIMIZATION" in result.output
        assert "BootstrapFewShot" in result.output

    def test_optimize_medium_budget(self, runner: CliRunner):
        result = runner.invoke(cli, ["optimize", "--budget", "medium"])
        assert result.exit_code == 0
        assert "MIPROv2" in result.output

    def test_optimize_heavy_budget(self, runner: CliRunner):
        result = runner.invoke(cli, ["optimize", "--budget", "heavy"])
        assert result.exit_code == 0
        assert "GEPA" in result.output

    def test_optimize_shows_note_without_datasets(self, runner: CliRunner):
        result = runner.invoke(cli, ["optimize", "--budget", "light"])
        assert result.exit_code == 0
        assert "Provide --trainset and --valset" in result.output


# -------------------------------------------------------------------------
# LLM-dependent commands: graceful error when no API key
# -------------------------------------------------------------------------


class TestLLMCommandsGracefulFailure:
    """Test that LLM-dependent commands fail gracefully with a clear error
    message when no API key is configured."""

    def test_extract_no_api_key(
        self, runner: CliRunner, tmp_text_file: Path, _no_api_key
    ):
        result = runner.invoke(
            cli, ["extract", "--document", str(tmp_text_file)]
        )
        assert result.exit_code != 0
        assert "API key" in result.output or "api_key" in result.output

    def test_extract_no_document(self, runner: CliRunner, _no_api_key):
        result = runner.invoke(cli, ["extract"])
        assert result.exit_code != 0
        assert "--document is required" in result.output

    def test_extract_missing_document(
        self, runner: CliRunner, tmp_path: Path, _no_api_key
    ):
        missing = tmp_path / "nonexistent.txt"
        result = runner.invoke(
            cli, ["extract", "--document", str(missing)]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_extract_bad_template(
        self, runner: CliRunner, tmp_text_file: Path, _no_api_key
    ):
        result = runner.invoke(
            cli, ["extract", "--document", str(tmp_text_file), "--template", "nonexistent_template"]
        )
        assert result.exit_code != 0
        assert "Template not found" in result.output

    def test_summarize_no_input(self, runner: CliRunner, _no_api_key):
        result = runner.invoke(cli, ["summarize"])
        assert result.exit_code != 0
        assert "Provide --document or --dir" in result.output

    def test_summarize_no_api_key(
        self, runner: CliRunner, tmp_text_file: Path, _no_api_key
    ):
        result = runner.invoke(
            cli, ["summarize", "--document", str(tmp_text_file)]
        )
        assert result.exit_code != 0
        assert "API key" in result.output or "api_key" in result.output

    def test_deidentify_llm_no_api_key(
        self, runner: CliRunner, tmp_text_file: Path, _no_api_key
    ):
        """Without --regex-only, deidentify requires LLM and should fail gracefully."""
        result = runner.invoke(
            cli, ["deidentify", "--document", str(tmp_text_file)]
        )
        assert result.exit_code != 0
        assert "API key" in result.output or "api_key" in result.output

    def test_schema_generate_no_api_key(self, runner: CliRunner, _no_api_key):
        result = runner.invoke(
            cli,
            ["schema", "generate", "--description", "A simple patient record"],
        )
        assert result.exit_code != 0
        assert "API key" in result.output or "api_key" in result.output


# -------------------------------------------------------------------------
# batch command validation
# -------------------------------------------------------------------------


class TestBatch:
    def test_batch_requires_input_dir(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(
            cli, ["batch", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "--input-dir is required" in result.output

    def test_batch_requires_output_dir(self, runner: CliRunner, tmp_path: Path):
        result = runner.invoke(
            cli, ["batch", "--input-dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "--output-dir is required" in result.output

    def test_batch_validates_input_dir_exists(self, runner: CliRunner, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        result = runner.invoke(
            cli,
            ["batch", "--input-dir", str(missing), "--output-dir", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_batch_shows_config_table(self, runner: CliRunner, tmp_path: Path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        result = runner.invoke(
            cli,
            [
                "batch",
                "--input-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--template", "chest_ct",
                "--workers", "2",
            ],
        )
        assert result.exit_code == 0
        assert "BATCH PROCESSING" in result.output
        assert "chest_ct" in result.output
        assert "Batch complete" in result.output


# -------------------------------------------------------------------------
# Version and help
# -------------------------------------------------------------------------


class TestVersionAndHelp:
    def test_version(self, runner: CliRunner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "2.0" in result.output

    def test_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MOSAICX" in result.output

    def test_deidentify_help_shows_regex_only(self, runner: CliRunner):
        result = runner.invoke(cli, ["deidentify", "--help"])
        assert result.exit_code == 0
        assert "--regex-only" in result.output

    def test_optimize_help_shows_budget_choices(self, runner: CliRunner):
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0
        assert "light" in result.output
        assert "medium" in result.output
        assert "heavy" in result.output

    def test_schema_generate_help_shows_description(self, runner: CliRunner):
        result = runner.invoke(cli, ["schema", "generate", "--help"])
        assert result.exit_code == 0
        assert "--description" in result.output

    def test_template_validate_help_shows_file(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "validate", "--help"])
        assert result.exit_code == 0
        assert "--file" in result.output

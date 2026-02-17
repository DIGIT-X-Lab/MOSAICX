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
        assert "Mode" in result.output
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
        assert "does not exist" in result.output

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
        """Regex-only mode works with a directory of files via batch."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        for i in range(3):
            p = input_dir / f"report_{i}.txt"
            p.write_text(f"Patient phone: (555) 000-{i:04d}\nFindings: normal.\n")

        result = runner.invoke(
            cli,
            ["deidentify", "--dir", str(input_dir), "--regex-only",
             "--output-dir", str(output_dir)],
        )
        assert result.exit_code == 0
        # Batch success message
        assert "3/3 succeeded" in result.output
        # Output JSON files should exist with scrubbed content
        for i in range(3):
            out_file = output_dir / f"report_{i}.json"
            assert out_file.exists(), f"Missing output: {out_file}"
            import json
            data = json.loads(out_file.read_text())
            assert "(555) 000-" not in data["redacted_text"]
            assert "normal" in data["redacted_text"]

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
        assert "does not exist" in result.output


# -------------------------------------------------------------------------
# optimize (no LLM needed for config display)
# -------------------------------------------------------------------------


class TestOptimize:
    def test_optimize_requires_pipeline(self, runner: CliRunner):
        result = runner.invoke(cli, ["optimize", "--budget", "light"])
        assert result.exit_code != 0
        assert "--pipeline is required" in result.output

    def test_optimize_requires_trainset(self, runner: CliRunner):
        result = runner.invoke(
            cli, ["optimize", "--pipeline", "extraction", "--budget", "light"]
        )
        assert result.exit_code != 0
        assert "--trainset is required" in result.output

    def test_optimize_list_pipelines(self, runner: CliRunner):
        result = runner.invoke(cli, ["optimize", "--list-pipelines"])
        assert result.exit_code == 0
        assert "extraction" in result.output.lower()


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
        assert "--document or --dir is required" in result.output

    def test_extract_missing_document(
        self, runner: CliRunner, tmp_path: Path, _no_api_key
    ):
        missing = tmp_path / "nonexistent.txt"
        result = runner.invoke(
            cli, ["extract", "--document", str(missing)]
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_extract_list_modes(self, runner: CliRunner):
        result = runner.invoke(cli, ["extract", "--list-modes"])
        assert result.exit_code == 0
        assert "radiology" in result.output
        assert "pathology" in result.output

    def test_extract_unknown_mode(self, runner: CliRunner, tmp_text_file: Path, _no_api_key):
        result = runner.invoke(
            cli, ["extract", "--document", str(tmp_text_file), "--mode", "nonexistent"]
        )
        assert result.exit_code != 0
        assert "Unknown mode" in result.output

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



# -------------------------------------------------------------------------
# batch command validation
# -------------------------------------------------------------------------


class TestExtractDir:
    """Tests for extract --dir (replaces old batch command)."""

    def test_extract_dir_shows_in_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--dir" in result.output

    def test_extract_document_and_dir_mutually_exclusive(
        self, runner: CliRunner, tmp_path: Path
    ):
        doc = tmp_path / "test.txt"
        doc.write_text("test")
        result = runner.invoke(
            cli, ["extract", "--document", str(doc), "--dir", str(tmp_path)]
        )
        assert result.exit_code != 0


# -------------------------------------------------------------------------
# schema list / refine
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Unified --template flag
# -------------------------------------------------------------------------


class TestUnifiedTemplateFlag:
    """Tests for the unified --template flag on extract and batch commands."""

    def test_extract_help_shows_template(self, runner: CliRunner):
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--template" in result.output

    def test_extract_help_shows_score(self, runner: CliRunner):
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--score" in result.output

    def test_extract_template_and_mode_mutually_exclusive(
        self, runner: CliRunner, tmp_text_file: Path
    ):
        """--template and --mode on extract are mutually exclusive."""
        result = runner.invoke(
            cli,
            [
                "extract",
                "--document", str(tmp_text_file),
                "--template", "chest_ct",
                "--mode", "radiology",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_extract_unknown_template_shows_error(
        self, runner: CliRunner, tmp_text_file: Path, _no_api_key
    ):
        """Unknown template name shows helpful error."""
        result = runner.invoke(
            cli,
            ["extract", "--document", str(tmp_text_file), "--template", "nonexistent_xyz"],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_extract_dir_help_shows_workers(self, runner: CliRunner):
        """extract --help shows --workers flag (batch functionality)."""
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--workers" in result.output



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

    def test_template_validate_help_shows_file(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "validate", "--help"])
        assert result.exit_code == 0
        assert "--file" in result.output


# -------------------------------------------------------------------------
# Phase 2: template create / show / refine
# -------------------------------------------------------------------------


class TestTemplateCreate:
    """Tests for the template create command."""

    def test_template_create_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "create", "--help"])
        assert result.exit_code == 0
        assert "--describe" in result.output
        assert "--from-document" in result.output
        assert "--from-url" in result.output
        assert "--from-json" in result.output

    def test_template_create_no_source_error(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "create"])
        assert result.exit_code != 0
        assert "Provide" in result.output

    def test_template_create_from_json(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """--from-json converts a SchemaSpec JSON to YAML template."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            json_file = tmp_path / "MySchema.json"
            json_file.write_text(
                '{"class_name":"MySchema","description":"A test schema",'
                '"fields":[{"name":"findings","type":"str","description":"findings","required":true,"enum_values":null},'
                '{"name":"severity","type":"enum","description":"severity level","required":false,'
                '"enum_values":["mild","moderate","severe"]}]}'
            )
            result = runner.invoke(
                cli,
                ["template", "create", "--from-json", str(json_file)],
            )
            assert result.exit_code == 0
            assert "Template created" in result.output
            assert "MySchema" in result.output

            # Verify YAML file was created
            yaml_path = tmp_path / "templates" / "MySchema.yaml"
            assert yaml_path.exists()
            content = yaml_path.read_text()
            assert "findings" in content
            assert "severity" in content
        finally:
            get_config.cache_clear()

    def test_template_create_from_json_with_name(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """--from-json with --name overrides the class name."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            json_file = tmp_path / "Original.json"
            json_file.write_text(
                '{"class_name":"Original","description":"test",'
                '"fields":[{"name":"x","type":"str","description":"x","required":true,"enum_values":null}]}'
            )
            result = runner.invoke(
                cli,
                ["template", "create", "--from-json", str(json_file), "--name", "Renamed"],
            )
            assert result.exit_code == 0
            assert "Renamed" in result.output
            yaml_path = tmp_path / "templates" / "Renamed.yaml"
            assert yaml_path.exists()
        finally:
            get_config.cache_clear()

    def test_template_create_from_json_with_output(self, runner: CliRunner, tmp_path: Path):
        """--from-json with --output saves to custom path."""
        json_file = tmp_path / "Test.json"
        json_file.write_text(
            '{"class_name":"Test","description":"test",'
            '"fields":[{"name":"x","type":"str","description":"x","required":true,"enum_values":null}]}'
        )
        output_path = tmp_path / "custom" / "out.yaml"
        result = runner.invoke(
            cli,
            ["template", "create", "--from-json", str(json_file), "--output", str(output_path)],
        )
        assert result.exit_code == 0
        assert output_path.exists()

    def test_template_create_from_json_missing_file(self, runner: CliRunner, tmp_path: Path):
        missing = tmp_path / "nonexistent.json"
        result = runner.invoke(
            cli,
            ["template", "create", "--from-json", str(missing)],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_template_create_from_json_invalid_json(self, runner: CliRunner, tmp_path: Path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not valid json at all")
        result = runner.invoke(
            cli,
            ["template", "create", "--from-json", str(bad_json)],
        )
        assert result.exit_code != 0
        assert "Invalid" in result.output

    def test_template_create_from_json_with_mode(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """--mode embeds pipeline mode in the YAML template."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            json_file = tmp_path / "Test.json"
            json_file.write_text(
                '{"class_name":"Test","description":"test",'
                '"fields":[{"name":"x","type":"str","description":"x","required":true,"enum_values":null}]}'
            )
            result = runner.invoke(
                cli,
                ["template", "create", "--from-json", str(json_file), "--mode", "radiology"],
            )
            assert result.exit_code == 0
            yaml_path = tmp_path / "templates" / "Test.yaml"
            content = yaml_path.read_text()
            assert "mode: radiology" in content
        finally:
            get_config.cache_clear()


class TestTemplateCreateFromRadReport:
    """Tests for the --from-radreport flag (no actual API calls)."""

    def test_from_radreport_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "create", "--help"])
        assert "--from-radreport" in result.output

    def test_from_radreport_no_source_includes_option(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "create"])
        assert result.exit_code != 0
        assert "from-radreport" in result.output


class TestTemplateShow:
    """Tests for the template show command."""

    def test_template_show_builtin(self, runner: CliRunner):
        """template show works for built-in templates."""
        result = runner.invoke(cli, ["template", "show", "chest_ct"])
        assert result.exit_code == 0
        # Section title is uppercased by Rich: "CHESTCTREPORT (BUILT-IN)"
        assert "CHESTCTREPORT" in result.output.upper()
        assert "BUILT-IN" in result.output.upper()

    def test_template_show_builtin_sections(self, runner: CliRunner):
        """template show shows sections for built-in templates."""
        result = runner.invoke(cli, ["template", "show", "brain_mri"])
        assert result.exit_code == 0
        assert "indication" in result.output
        assert "impression" in result.output

    def test_template_show_user_template(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """template show works for user-created templates."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            tpl_dir = tmp_path / "templates"
            tpl_dir.mkdir()
            (tpl_dir / "my_tpl.yaml").write_text(
                "name: MyTemplate\n"
                "description: Test template\n"
                "mode: radiology\n"
                "sections:\n"
                "  - name: findings\n"
                "    type: str\n"
                "    required: true\n"
                "    description: Clinical findings\n"
            )
            result = runner.invoke(cli, ["template", "show", "my_tpl"])
            assert result.exit_code == 0
            assert "MYTEMPLATE" in result.output.upper()
            assert "USER" in result.output.upper()
            assert "findings" in result.output
        finally:
            get_config.cache_clear()

    def test_template_show_saved_schema_fallback(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """template show falls back to saved schemas."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir()
            (schema_dir / "OldSchema.json").write_text(
                '{"class_name":"OldSchema","description":"Legacy schema",'
                '"fields":[{"name":"x","type":"str","description":"x","required":true,"enum_values":null}]}'
            )
            result = runner.invoke(cli, ["template", "show", "OldSchema"])
            assert result.exit_code == 0
            assert "OLDSCHEMA" in result.output.upper()
            assert "LEGACY SCHEMA" in result.output.upper()
            assert "template create --from-json" in result.output
        finally:
            get_config.cache_clear()

    def test_template_show_not_found(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """template show with unknown name shows error."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            result = runner.invoke(cli, ["template", "show", "nonexistent"])
            assert result.exit_code != 0
            assert "not found" in result.output.lower()
        finally:
            get_config.cache_clear()


class TestTemplateRefineHelp:
    """Tests for the template refine command (non-LLM tests only)."""

    def test_template_refine_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "refine", "--help"])
        assert result.exit_code == 0
        assert "--instruction" in result.output

    def test_template_refine_not_found(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            result = runner.invoke(
                cli,
                ["template", "refine", "nonexistent", "--instruction", "add a field"],
            )
            assert result.exit_code != 0
            assert "not found" in result.output.lower()
        finally:
            get_config.cache_clear()


# -------------------------------------------------------------------------
# template migrate
# -------------------------------------------------------------------------


class TestTemplateMigrate:
    """Tests for the template migrate command."""

    def test_migrate_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "migrate", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output

    def test_migrate_no_schemas(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """No schemas directory -- nothing to migrate."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            result = runner.invoke(cli, ["template", "migrate"])
            assert result.exit_code == 0
            assert "nothing to migrate" in result.output.lower()
        finally:
            get_config.cache_clear()

    def test_migrate_empty_schemas_dir(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """Empty schemas directory -- nothing to migrate."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            (tmp_path / "schemas").mkdir()
            result = runner.invoke(cli, ["template", "migrate"])
            assert result.exit_code == 0
            assert "nothing to migrate" in result.output.lower()
        finally:
            get_config.cache_clear()

    def test_migrate_dry_run(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """Dry run shows what would be migrated without writing files."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir()
            (schema_dir / "TestModel.json").write_text(
                '{"class_name":"TestModel","description":"A test",'
                '"fields":[{"name":"x","type":"str","description":"x",'
                '"required":true,"enum_values":null}]}'
            )
            result = runner.invoke(cli, ["template", "migrate", "--dry-run"])
            assert result.exit_code == 0
            assert "would migrate" in result.output.lower()
            # Should NOT create the YAML file
            assert not (tmp_path / "templates" / "TestModel.yaml").exists()
        finally:
            get_config.cache_clear()

    def test_migrate_creates_yaml(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """Migration creates YAML files from JSON schemas."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir()
            (schema_dir / "TestModel.json").write_text(
                '{"class_name":"TestModel","description":"A test",'
                '"fields":[{"name":"x","type":"str","description":"x",'
                '"required":true,"enum_values":null}]}'
            )
            result = runner.invoke(cli, ["template", "migrate"])
            assert result.exit_code == 0
            assert "migrated" in result.output.lower()

            yaml_path = tmp_path / "templates" / "TestModel.yaml"
            assert yaml_path.exists()
            content = yaml_path.read_text()
            assert "TestModel" in content
            assert "sections" in content
        finally:
            get_config.cache_clear()

    def test_migrate_skips_existing_yaml(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """Migration skips schemas that already have YAML templates."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir()
            (schema_dir / "TestModel.json").write_text(
                '{"class_name":"TestModel","description":"Test",'
                '"fields":[{"name":"x","type":"str","description":"x",'
                '"required":true,"enum_values":null}]}'
            )
            # Pre-create the YAML file
            tpl_dir = tmp_path / "templates"
            tpl_dir.mkdir()
            (tpl_dir / "TestModel.yaml").write_text("existing content")

            result = runner.invoke(cli, ["template", "migrate"])
            assert result.exit_code == 0
            assert "skip" in result.output.lower()
            # Should NOT overwrite
            assert (tpl_dir / "TestModel.yaml").read_text() == "existing content"
        finally:
            get_config.cache_clear()


# -------------------------------------------------------------------------
# template versioning (history, revert, diff)
# -------------------------------------------------------------------------


SAMPLE_TPL_V1 = (
    "name: TestTpl\n"
    "description: version 1\n"
    "mode: radiology\n"
    "sections:\n"
    "  - name: finding_a\n"
    "    type: str\n"
    "    required: true\n"
)

SAMPLE_TPL_V2 = (
    "name: TestTpl\n"
    "description: version 2\n"
    "mode: radiology\n"
    "sections:\n"
    "  - name: finding_a\n"
    "    type: str\n"
    "    required: true\n"
    "  - name: finding_b\n"
    "    type: str\n"
    "    required: false\n"
    "    description: New field in v2\n"
)


class TestTemplateVersioning:
    """Tests for template history, revert, and diff commands."""

    def _setup_versioned_template(self, tmp_path: Path):
        """Create a user template with one archived version."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir(parents=True)
        (tpl_dir / "TestTpl.yaml").write_text(SAMPLE_TPL_V2)

        # Create history
        history_dir = tpl_dir / ".history"
        history_dir.mkdir()
        (history_dir / "TestTpl_v1.yaml").write_text(SAMPLE_TPL_V1)

    def test_history_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["template", "history", "--help"])
        assert result.exit_code == 0

    def test_history_not_found(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            result = runner.invoke(cli, ["template", "history", "nonexistent"])
            assert result.exit_code != 0
            assert "not found" in result.output.lower()
        finally:
            get_config.cache_clear()

    def test_history_shows_versions(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            self._setup_versioned_template(tmp_path)
            result = runner.invoke(cli, ["template", "history", "TestTpl"])
            assert result.exit_code == 0
            assert "v1" in result.output
            assert "current" in result.output.lower()
        finally:
            get_config.cache_clear()

    def test_history_no_versions(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            tpl_dir = tmp_path / "templates"
            tpl_dir.mkdir()
            (tpl_dir / "Fresh.yaml").write_text(SAMPLE_TPL_V1)
            result = runner.invoke(cli, ["template", "history", "Fresh"])
            assert result.exit_code == 0
            assert "no version history" in result.output.lower()
        finally:
            get_config.cache_clear()

    def test_diff_shows_changes(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            self._setup_versioned_template(tmp_path)
            result = runner.invoke(
                cli, ["template", "diff", "TestTpl", "--version", "1"]
            )
            assert result.exit_code == 0
            assert "finding_b" in result.output  # added in v2
        finally:
            get_config.cache_clear()

    def test_diff_version_not_found(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            self._setup_versioned_template(tmp_path)
            result = runner.invoke(
                cli, ["template", "diff", "TestTpl", "--version", "99"]
            )
            assert result.exit_code != 0
            assert "not found" in result.output.lower()
        finally:
            get_config.cache_clear()

    def test_revert_restores_old_version(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            self._setup_versioned_template(tmp_path)
            result = runner.invoke(
                cli, ["template", "revert", "TestTpl", "--version", "1"]
            )
            assert result.exit_code == 0
            assert "reverted" in result.output.lower()

            # Current should now be v1 content
            current = (tmp_path / "templates" / "TestTpl.yaml").read_text()
            assert "version 1" in current
            assert "finding_b" not in current

            # Should have archived v2 as v2
            assert (tmp_path / "templates" / ".history" / "TestTpl_v2.yaml").exists()
        finally:
            get_config.cache_clear()

    def test_revert_version_not_found(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            self._setup_versioned_template(tmp_path)
            result = runner.invoke(
                cli, ["template", "revert", "TestTpl", "--version", "99"]
            )
            assert result.exit_code != 0
            assert "not found" in result.output.lower()
        finally:
            get_config.cache_clear()

"""
Enhanced CLI tests exercising direct command invocation and workflow flows.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mosaicx.mosaicx import cli, extract, generate


class TestCLIInterface:
    """Additional coverage for top-level CLI behaviour."""

    def test_cli_help_mentions_commands(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Usage" in result.output
        assert "extract" in result.output
        assert "generate" in result.output

    def test_cli_version_includes_full_string(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "MOSAICX, version" in result.output

    @patch("mosaicx.cli.app.show_main_banner")
    def test_cli_no_command_triggers_banner(self, mock_banner: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [])

        mock_banner.assert_called_once()
        assert "Welcome to MOSAICX" in result.output
        assert result.exit_code == 0


class TestGenerateCommand:
    """Direct command invocation tests for generate command."""

    def test_generate_help_direct(self) -> None:
        runner = CliRunner()
        result = runner.invoke(generate, ["--help"])

        assert result.exit_code == 0
        assert "Generate Pydantic schemas" in result.output
        assert "--desc" in result.output

    @patch("mosaicx.cli.app.register_schema")
    @patch("mosaicx.cli.app.generate_schema")
    def test_generate_basic_functionality(
        self,
        mock_generate_schema: MagicMock,
        mock_register: MagicMock,
        temp_dir: Path,
    ) -> None:
        generated = MagicMock()
        generated.code = "class Generated: ..."
        generated.suggested_filename = "generated.py"
        generated.write.return_value = temp_dir / "generated.py"
        mock_generate_schema.return_value = generated
        mock_register.return_value = "schema-001"

        runner = CliRunner()
        result = runner.invoke(
            generate,
            [
                "--desc",
                "A simple model",
                "--class-name",
                "Generated",
            ],
        )

        assert result.exit_code == 0
        mock_generate_schema.assert_called_once()
        mock_register.assert_called_once()

    @patch("mosaicx.cli.app.generate_schema")
    def test_generate_supports_custom_model_direct(self, mock_generate_schema: MagicMock) -> None:
        generated = MagicMock()
        generated.code = "class Generated: ..."
        generated.suggested_filename = "generated.py"
        generated.write.return_value = Path("generated.py")
        mock_generate_schema.return_value = generated

        runner = CliRunner()
        runner.invoke(
            generate,
            [
                "--desc",
                "Schema",
                "--model",
                "llama3",
            ],
        )

        assert mock_generate_schema.call_args.kwargs["model"] == "llama3"

    @patch("mosaicx.cli.app.register_schema")
    @patch("mosaicx.cli.app.generate_schema")
    def test_generate_template_flag(
        self,
        mock_generate_schema: MagicMock,
        mock_register: MagicMock,
    ) -> None:
        generated = MagicMock()
        generated.code = "class Generated: ..."
        generated.suggested_filename = "generated.py"
        generated.write.return_value = Path("mosaicx/schema/templates/python/generated.py")
        mock_generate_schema.return_value = generated
        mock_register.return_value = "schema-002"

        runner = CliRunner()
        result = runner.invoke(
            generate,
            [
                "--desc",
                "Template schema",
                "--template",
            ],
        )

        assert result.exit_code == 0
        generated.write.assert_called_once()
        args, kwargs = generated.write.call_args
        assert args == (None,)
        assert kwargs == {"template": True}


class TestExtractCommand:
    """Direct command invocation tests for extract command."""

    def test_extract_help_direct(self) -> None:
        runner = CliRunner()
        result = runner.invoke(extract, ["--help"])

        assert result.exit_code == 0
        assert "Extract structured data from a clinical document" in result.output
        assert "--document" in result.output

    @patch("mosaicx.cli.app.extract_pdf")
    @patch("mosaicx.cli.app.list_schemas")
    @patch("mosaicx.cli.app.resolve_schema_reference")
    def test_extract_basic_flow(
        self,
        mock_resolve: MagicMock,
        mock_list: MagicMock,
        mock_extract_pdf: MagicMock,
        temp_dir: Path,
    ) -> None:
        pdf_file = temp_dir / "report.pdf"
        pdf_file.write_text("content")
        schema_path = temp_dir / "schema.py"
        schema_path.write_text("class Generated: ...")

        mock_resolve.return_value = schema_path
        mock_list.return_value = [
            {"file_path": str(schema_path), "class_name": "Generated"}
        ]
        fake_result = MagicMock()
        fake_result.record.model_dump.return_value = {"field": "value"}
        fake_result.record.model_dump_json.return_value = json.dumps({"field": "value"})
        mock_extract_pdf.return_value = fake_result

        runner = CliRunner()
        result = runner.invoke(
            extract,
            [
                "--document",
                str(pdf_file),
                "--schema",
                "generated",
            ],
        )

        assert result.exit_code == 0
        mock_resolve.assert_called_once_with("generated")
        mock_extract_pdf.assert_called_once()

    @patch("mosaicx.cli.app.extract_pdf")
    @patch("mosaicx.cli.app.list_schemas")
    @patch("mosaicx.cli.app.resolve_schema_reference")
    def test_extract_allows_json_save(
        self,
        mock_resolve: MagicMock,
        mock_list: MagicMock,
        mock_extract_pdf: MagicMock,
        temp_dir: Path,
    ) -> None:
        pdf_file = temp_dir / "report.pdf"
        pdf_file.write_text("content")
        schema_path = temp_dir / "schema.py"
        schema_path.write_text("class Generated: ...")

        mock_resolve.return_value = schema_path
        mock_list.return_value = [
            {"file_path": str(schema_path), "class_name": "Generated"}
        ]
        fake_result = MagicMock()
        fake_result.record.model_dump.return_value = {"field": "value"}
        fake_result.record.model_dump_json.return_value = json.dumps({"field": "value"})
        fake_result.write_json.return_value = temp_dir / "out.json"
        mock_extract_pdf.return_value = fake_result

        output_path = temp_dir / "saved.json"

        runner = CliRunner()
        result = runner.invoke(
            extract,
            [
                "--document",
                str(pdf_file),
                "--schema",
                "generated",
                "--save",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        fake_result.write_json.assert_called_once_with(output_path)
        assert output_path.exists()


class TestWorkflowIntegration:
    """Integration-style test combining generate and extract commands."""

    @patch("mosaicx.cli.app.extract_pdf")
    @patch("mosaicx.cli.app.list_schemas", return_value=[{"file_path": "schema.py", "class_name": "Generated"}])
    @patch("mosaicx.cli.app.resolve_schema_reference", return_value=Path("schema.py"))
    @patch("mosaicx.cli.app.register_schema", return_value="schema-001")
    @patch("mosaicx.cli.app.generate_schema")
    def test_generate_then_extract_workflow(
        self,
        mock_generate_schema: MagicMock,
        _: MagicMock,
        __: MagicMock,
        ___: MagicMock,
        mock_extract_pdf: MagicMock,
        temp_dir: Path,
    ) -> None:
        generated = MagicMock()
        generated.code = "class Generated: ..."
        generated.suggested_filename = "generated.py"
        generated.write.return_value = temp_dir / "generated.py"
        mock_generate_schema.return_value = generated

        fake_result = MagicMock()
        fake_result.record.model_dump.return_value = {"field": "value"}
        fake_result.record.model_dump_json.return_value = json.dumps({"field": "value"})
        mock_extract_pdf.return_value = fake_result

        runner = CliRunner()
        with runner.isolated_filesystem():
            gen_result = runner.invoke(
                cli,
                [
                    "generate",
                    "--desc",
                    "Workflow schema",
                ],
            )
            pdf_file = Path("report.pdf")
            pdf_file.write_text("content")
            ext_result = runner.invoke(
                cli,
                [
                    "extract",
                    "--document",
                    str(pdf_file),
                    "--schema",
                    "schema-001",
                ],
            )

        assert gen_result.exit_code == 0
        assert ext_result.exit_code == 0
        mock_generate_schema.assert_called_once()
        mock_extract_pdf.assert_called_once()

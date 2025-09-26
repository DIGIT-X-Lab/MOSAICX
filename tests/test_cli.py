"""
Test CLI Module

Tests for command-line interface functionality and user interactions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mosaicx.mosaicx import cli, extract, generate


class TestCLIInterface:
    """Test cases for CLI command interface."""

    def test_cli_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "MOSAICX" in result.output
        assert "generate" in result.output
        assert "summarize" in result.output

    def test_cli_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "1.0.8" in result.output

    def test_cli_verbose_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose"])

        assert result.exit_code == 0

    def test_cli_no_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        assert "Welcome to MOSAICX" in result.output


class TestGenerateCommand:
    """Test cases for generate command."""

    def test_generate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])

        assert result.exit_code == 0
        assert "Generate Pydantic schemas" in result.output
        assert "--schema-path" in result.output

    @patch("mosaicx.cli.app.register_schema")
    @patch("mosaicx.cli.app.generate_schema")
    def test_generate_success(self, mock_generate_schema: MagicMock, mock_register: MagicMock) -> None:
        generated = MagicMock()
        generated.code = "class TestModel: ..."
        generated.suggested_filename = "TestModel.py"
        generated.write.return_value = Path("TestModel.py")
        generated.class_name = "TestModel"
        generated.description = "desc"
        mock_generate_schema.return_value = generated
        mock_register.return_value = "schema-id"

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "--desc",
                    "Test model",
                    "--class-name",
                    "TestModel",
                ],
            )

        assert result.exit_code == 0
        mock_generate_schema.assert_called_once()
        generated.write.assert_called_once()
        mock_register.assert_called_once()

    @patch("mosaicx.cli.app.generate_schema")
    def test_generate_with_custom_model(self, mock_generate_schema: MagicMock) -> None:
        generated = MagicMock()
        generated.code = "class TestModel: ..."
        generated.suggested_filename = "TestModel.py"
        generated.write.return_value = Path("TestModel.py")
        mock_generate_schema.return_value = generated

        runner = CliRunner()
        with runner.isolated_filesystem():
            runner.invoke(
                cli,
                [
                    "generate",
                    "--desc",
                    "Test schema",
                    "--model",
                    "mistral:latest",
                ],
            )

        called_kwargs = mock_generate_schema.call_args.kwargs
        assert called_kwargs["model"] == "mistral:latest"

    def test_generate_missing_desc(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["generate"])

        assert result.exit_code != 0
        assert "Missing option" in result.output

    @patch("mosaicx.cli.app.generate_schema")
    def test_generate_save_to_file(self, mock_generate_schema: MagicMock) -> None:
        generated = MagicMock()
        generated.code = "class TestModel: ..."
        generated.suggested_filename = "TestModel.py"
        generated.write.return_value = Path("custom.py")
        mock_generate_schema.return_value = generated

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "--desc",
                    "Test schema",
                    "--schema-path",
                    "custom.py",
                ],
            )
            saved_path = Path("custom.py")

        assert result.exit_code == 0
        generated.write.assert_called_once_with(Path("custom.py"))
        assert saved_path.exists()

    @patch("mosaicx.cli.app.generate_schema", side_effect=RuntimeError("LLM failure"))
    def test_generate_failure(self, _: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate",
                "--desc",
                "Test schema",
            ],
        )

        assert result.exit_code != 0
        assert "Schema generation failed" in result.output


class TestExtractCommand:
    """Test cases for extract command."""

    def test_extract_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "--help"])

        assert result.exit_code == 0
        assert "Extract structured data from PDF" in result.output
        assert "--pdf" in result.output
        assert "--schema" in result.output

    def test_extract_missing_required_args(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["extract"])

        assert result.exit_code != 0
        assert "Missing option" in result.output

    @patch("mosaicx.cli.app.extract_pdf")
    @patch("mosaicx.cli.app.list_schemas")
    @patch("mosaicx.cli.app.resolve_schema_reference")
    def test_extract_success(
        self,
        mock_resolve: MagicMock,
        mock_list: MagicMock,
        mock_extract_pdf: MagicMock,
        temp_dir: Path,
    ) -> None:
        pdf_file = temp_dir / "sample.pdf"
        pdf_file.write_text("content")
        schema_path = temp_dir / "schema.py"
        schema_path.write_text("class TestModel: ...")

        mock_resolve.return_value = schema_path
        mock_list.return_value = [
            {
                "file_path": str(schema_path),
                "class_name": "TestModel",
            }
        ]
        extraction_result = MagicMock()
        extraction_result.record.model_dump.return_value = {"field": "value"}
        extraction_result.record.model_dump_json.return_value = json.dumps({"field": "value"})
        extraction_result.write_json.return_value = temp_dir / "result.json"
        mock_extract_pdf.return_value = extraction_result

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "extract",
                "--pdf",
                str(pdf_file),
                "--schema",
                "schema-id",
            ],
        )

        assert result.exit_code == 0
        mock_resolve.assert_called_once_with("schema-id")
        mock_extract_pdf.assert_called_once()

    @patch("mosaicx.cli.app.extract_pdf")
    @patch("mosaicx.cli.app.list_schemas")
    @patch("mosaicx.cli.app.resolve_schema_reference")
    def test_extract_with_save_option(
        self,
        mock_resolve: MagicMock,
        mock_list: MagicMock,
        mock_extract_pdf: MagicMock,
        temp_dir: Path,
    ) -> None:
        pdf_file = temp_dir / "sample.pdf"
        pdf_file.write_text("content")
        schema_path = temp_dir / "schema.py"
        schema_path.write_text("class TestModel: ...")

        mock_resolve.return_value = schema_path
        mock_list.return_value = [
            {
                "file_path": str(schema_path),
                "class_name": "TestModel",
            }
        ]
        extraction_result = MagicMock()
        extraction_result.record.model_dump.return_value = {"field": "value"}
        extraction_result.record.model_dump_json.return_value = json.dumps({"field": "value"})
        extraction_result.write_json.return_value = temp_dir / "result.json"
        mock_extract_pdf.return_value = extraction_result

        output_path = temp_dir / "output.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "extract",
                "--pdf",
                str(pdf_file),
                "--schema",
                "schema-id",
                "--save",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        extraction_result.write_json.assert_called_once_with(output_path)
        assert output_path.exists()

    @patch("mosaicx.cli.app.resolve_schema_reference", return_value=None)
    def test_extract_schema_not_found(self, _: MagicMock, temp_dir: Path) -> None:
        pdf_file = temp_dir / "sample.pdf"
        pdf_file.write_text("content")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "extract",
                "--pdf",
                str(pdf_file),
                "--schema",
                "missing",
            ],
        )

        assert result.exit_code != 0
        assert "Could not find schema" in result.output

    @patch("mosaicx.cli.app.extract_pdf")
    @patch("mosaicx.cli.app.list_schemas")
    @patch("mosaicx.cli.app.resolve_schema_reference")
    def test_extract_with_custom_model(
        self,
        mock_resolve: MagicMock,
        mock_list: MagicMock,
        mock_extract_pdf: MagicMock,
        temp_dir: Path,
    ) -> None:
        pdf_file = temp_dir / "sample.pdf"
        pdf_file.write_text("content")
        schema_path = temp_dir / "schema.py"
        schema_path.write_text("class TestModel: ...")

        mock_resolve.return_value = schema_path
        mock_list.return_value = [
            {
                "file_path": str(schema_path),
                "class_name": "TestModel",
            }
        ]
        extraction_result = MagicMock()
        extraction_result.record.model_dump.return_value = {"field": "value"}
        extraction_result.record.model_dump_json.return_value = json.dumps({"field": "value"})
        mock_extract_pdf.return_value = extraction_result

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "extract",
                "--pdf",
                str(pdf_file),
                "--schema",
                "schema-id",
                "--model",
                "mistral:latest",
            ],
        )

        call_kwargs = mock_extract_pdf.call_args.kwargs
        assert call_kwargs["model"] == "mistral:latest"

    @patch("mosaicx.cli.app.extract_pdf", side_effect=RuntimeError("Extraction failed"))
    @patch("mosaicx.cli.app.list_schemas", return_value=[{"file_path": "schema.py", "class_name": "TestModel"}])
    @patch("mosaicx.cli.app.resolve_schema_reference", return_value=Path("schema.py"))
    def test_extract_failure(
        self,
        _: MagicMock,
        __: MagicMock,
        ___: MagicMock,
        temp_dir: Path,
    ) -> None:
        pdf_file = temp_dir / "sample.pdf"
        pdf_file.write_text("content")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "extract",
                "--pdf",
                str(pdf_file),
                "--schema",
                "schema-id",
            ],
        )

        assert result.exit_code != 0
        assert "Unexpected error" in result.output


class TestSchemaResolution:
    """Test cases for schema reference resolution."""

    @patch("mosaicx.utils.pathing.get_schema_by_id")
    def test_resolve_schema_by_id(self, mock_get_schema: MagicMock) -> None:
        from mosaicx.utils import resolve_schema_reference

        mock_get_schema.return_value = {"file_path": "/path/to/schema.py"}

        result = resolve_schema_reference("schema_id_001")

        assert result == Path("/path/to/schema.py")
        mock_get_schema.assert_called_once_with("schema_id_001")

    def test_resolve_schema_by_filename(self) -> None:
        from mosaicx.utils import resolve_schema_reference

        with patch("pathlib.Path.exists", return_value=True):
            result = resolve_schema_reference("test_schema.py")

        assert result is not None
        assert result.name == "test_schema.py"

    def test_resolve_schema_by_path(self) -> None:
        from mosaicx.utils import resolve_schema_reference

        with patch("pathlib.Path.exists", return_value=True):
            result = resolve_schema_reference("/full/path/to/schema.py")

        assert result == Path("/full/path/to/schema.py")

    @patch("mosaicx.utils.pathing.get_schema_by_id", return_value=None)
    def test_resolve_schema_not_found(self, _: MagicMock) -> None:
        from mosaicx.utils import resolve_schema_reference

        with patch("pathlib.Path.exists", return_value=False):
            result = resolve_schema_reference("missing")

        assert result is None


class TestVerboseOutput:
    """Test cases for verbose output functionality."""

    @patch("mosaicx.cli.app.generate_schema")
    def test_generate_verbose_output(self, mock_generate_schema: MagicMock) -> None:
        generated = MagicMock()
        generated.code = "class TestModel: ..."
        generated.suggested_filename = "TestModel.py"
        generated.write.return_value = Path("TestModel.py")
        mock_generate_schema.return_value = generated

        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli,
                [
                    "--verbose",
                    "generate",
                    "--desc",
                    "Test schema",
                ],
            )

        assert result.exit_code == 0
        assert "Generating schema" in result.output

    @patch("mosaicx.cli.app.extract_pdf")
    @patch("mosaicx.cli.app.list_schemas")
    @patch("mosaicx.cli.app.resolve_schema_reference")
    def test_extract_verbose_output(
        self,
        mock_resolve: MagicMock,
        mock_list: MagicMock,
        mock_extract_pdf: MagicMock,
        temp_dir: Path,
    ) -> None:
        pdf_file = temp_dir / "sample.pdf"
        pdf_file.write_text("content")
        schema_path = temp_dir / "schema.py"
        schema_path.write_text("class TestModel: ...")

        mock_resolve.return_value = schema_path
        mock_list.return_value = [
            {
                "file_path": str(schema_path),
                "class_name": "TestModel",
            }
        ]
        extraction_result = MagicMock()
        extraction_result.record.model_dump.return_value = {"field": "value"}
        extraction_result.record.model_dump_json.return_value = json.dumps({"field": "value"})
        mock_extract_pdf.return_value = extraction_result

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--verbose",
                "extract",
                "--pdf",
                str(pdf_file),
                "--schema",
                "schema-id",
            ],
        )

        assert result.exit_code == 0
        assert "Extracting from" in result.output
        assert "Using schema" in result.output
        assert "Using model" in result.output

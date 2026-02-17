# tests/test_cli.py
"""Tests for the CLI skeleton."""

import pytest
from click.testing import CliRunner


class TestCLISkeleton:

    def test_cli_group_exists(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_extract_command_registered(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "document" in result.output.lower()

    def test_template_command_registered(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["template", "--help"])
        assert result.exit_code == 0

    def test_schema_command_removed(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["schema", "--help"])
        assert result.exit_code != 0  # schema group removed in Phase 3

    def test_summarize_command_registered(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--help"])
        assert result.exit_code == 0

    def test_deidentify_command_registered(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["deidentify", "--help"])
        assert result.exit_code == 0

    def test_optimize_command_registered(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0

    def test_config_command_registered(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0

    def test_version_flag(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "2.0" in result.output

"""Tests for the mosaicx doctor CLI command."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner


@pytest.mark.unit
class TestDoctorCommand:
    def test_doctor_runs_without_error(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        # May exit 1 if no LLM backend running, that's fine
        assert result.exit_code in (0, 1)
        assert "DOCTOR" in result.output.upper() or "doctor" in result.output.lower()

    def test_doctor_json_output(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        assert result.exit_code in (0, 1)
        # Find the JSON in output (banner text may appear before it)
        output = result.output
        json_start = output.find("{")
        if json_start >= 0:
            json_text = output[json_start:]
            data = json.loads(json_text)
            assert "checks" in data
            assert "summary" in data

    def test_doctor_checks_python(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert "Python" in result.output or "python" in result.output

    def test_doctor_fix_flag_exists(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--fix"])
        assert result.exit_code in (0, 1)

"""Tests for the mosaicx setup CLI command."""
from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.mark.unit
class TestSetupCommand:
    def test_setup_command_exists(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0
        assert "setup" in result.output.lower()

    def test_setup_non_interactive_runs(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--non-interactive"])
        assert result.exit_code in (0, 1)

    def test_setup_detects_platform(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--non-interactive"])
        output_lower = result.output.lower()
        assert any(
            p in output_lower
            for p in ["apple silicon", "dgx spark", "linux", "macos", "intel mac"]
        )

    def test_setup_full_flag_exists(self):
        from mosaicx.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["setup", "--help"])
        assert "--full" in result.output


@pytest.mark.unit
class TestRecommendModel:
    def test_mac_low_ram(self):
        from mosaicx.setup import recommend_model
        assert "20b" in recommend_model("macos-arm64", 32.0).lower()

    def test_mac_high_ram(self):
        from mosaicx.setup import recommend_model
        assert "120b" in recommend_model("macos-arm64", 128.0).lower()

    def test_dgx_default(self):
        from mosaicx.setup import recommend_model
        assert "120b" in recommend_model("dgx-spark", 128.0).lower()

    def test_linux_low_ram(self):
        from mosaicx.setup import recommend_model
        assert "20b" in recommend_model("linux-x86_64", 16.0).lower()

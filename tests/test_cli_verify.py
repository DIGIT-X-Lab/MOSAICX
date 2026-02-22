from __future__ import annotations

from click.testing import CliRunner


class TestVerifyCommand:
    def test_verify_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["verify", "--help"])
        assert result.exit_code == 0
        assert "source" in result.output.lower()

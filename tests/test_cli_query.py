from __future__ import annotations

import pytest
from click.testing import CliRunner


class TestQueryCommand:
    def test_query_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0

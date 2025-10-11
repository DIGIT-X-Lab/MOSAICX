from __future__ import annotations

from unittest.mock import patch

from rich.console import Console

from mosaicx.display import console, show_main_banner, styled_message


def test_console_instance() -> None:
    assert isinstance(console, Console)


def test_show_main_banner_calls_console() -> None:
    with patch("mosaicx.display.console") as mock_console:
        show_main_banner()
        assert mock_console.print.call_count >= 1


def test_styled_message_uses_console() -> None:
    with patch("mosaicx.display.console") as mock_console:
        styled_message("hello", "info")
        mock_console.print.assert_called_once()
        assert "hello" in str(mock_console.print.call_args[0][0])

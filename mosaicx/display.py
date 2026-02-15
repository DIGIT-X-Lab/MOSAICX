# mosaicx/display.py
"""Terminal display utilities using Rich."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

MOSAICX_COLORS = {
    "primary": "#ff79c6",
    "secondary": "#6272a4",
    "success": "#50fa7b",
    "warning": "#f1fa8c",
    "error": "#ff5555",
    "info": "#8be9fd",
    "accent": "#bd93f9",
    "muted": "#44475a",
}


def show_main_banner() -> None:
    """Display the MOSAICX banner."""
    title = Text("MOSAICX", style=f"bold {MOSAICX_COLORS['primary']}")
    subtitle = Text("Medical Document Structuring Platform", style=MOSAICX_COLORS['secondary'])
    panel = Panel(
        Text.assemble(title, "\n", subtitle),
        border_style=MOSAICX_COLORS["accent"],
        padding=(1, 4),
    )
    console.print(panel)


def styled_message(message: str, style: str = "info") -> None:
    """Print a styled message."""
    color = MOSAICX_COLORS.get(style, MOSAICX_COLORS["info"])
    console.print(message, style=color)

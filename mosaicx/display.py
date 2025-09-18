"""
MOSAICX Display Module - Terminal User Interface Components

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Overview:
---------
This module provides sophisticated terminal display capabilities for the MOSAICX
application, leveraging modern Python libraries to create visually appealing and
informative command-line interfaces. It serves as the primary presentation layer
for user interactions and system feedback.

Core Functionality:
------------------
• Branded application banners with gradient styling
• Institutional attribution and contact information display  
• Status messages with semantic color coding
• Progress indicators and loading animations
• Interactive console elements with Rich library integration

Architecture:
------------
The module follows a functional design pattern, exposing utility functions for
different display scenarios while maintaining a centralized console instance for
consistent styling across the application.

Usage Examples:
--------------
Display main application banner:
    >>> from mosaicx.display import show_main_banner
    >>> show_main_banner()

Custom styled console output:
    >>> from mosaicx.display import console, styled_message
    >>> console.print("Processing complete", style="bold green")
    >>> styled_message("Warning: Check configuration", "warning")

Dependencies:
------------
External Libraries:
    • cfonts (^1.0.0): ASCII art text generation and styling
    • rich (^13.0.0): Advanced terminal formatting, tables, and progress bars

Standard Library:
    • typing: Type hint support for better code documentation

Module Metadata:
---------------
Author:        Lalith Kumar Shiyam Sundar, PhD
Email:         Lalith.shiyam@med.uni-muenchen.de  
Institution:   DIGIT-X Lab, LMU Radiology | LMU University Hospital
License:       AGPL-3.0 (GNU Affero General Public License v3.0)
Version:       1.0.0
Created:       2025-09-18
Last Modified: 2025-09-18

Copyright Notice:
----------------
© 2025 DIGIT-X Lab, LMU Radiology | LMU University Hospital
This software is distributed under the AGPL-3.0 license.
See LICENSE file for full terms and conditions.
"""

from typing import Optional, Dict, Any
from cfonts import say
from rich.console import Console
from rich.text import Text
from rich.align import Align
from rich.panel import Panel

# Module metadata
__version__ = "1.0.0"
__author__ = "Lalith Kumar Shiyam Sundar, PhD"
__email__ = "Lalith.shiyam@med.uni-muenchen.de"
__institution__ = "DIGIT-X Lab, LMU Radiology | LMU University Hospital"
__license__ = "AGPL-3.0"
__copyright__ = "© 2025 DIGIT-X Lab, LMU Radiology | LMU University Hospital"

# Color scheme for consistent branding
MOSAICX_COLORS = {
    "primary": "#FC683F",      # Orange accent
    "secondary": "#B6B6B6",    # Light gray
    "success": "green",
    "warning": "yellow", 
    "error": "red",
    "info": "blue"
}

# Global console instance for consistent styling
console = Console()

# Export list for public API
__all__ = [
    "console", 
    "show_main_banner", 
    "styled_message", 
    "MOSAICX_COLORS",
    "__version__",
    "__author__"
]


def show_main_banner() -> None:
    """
    Display the main MOSAICX application banner with institutional branding.
    
    This function creates a visually appealing startup banner featuring:
    - Large ASCII art title using cfonts
    - Full application name expansion
    - Institutional attribution
    - Development status teaser
    """
    # Clear screen space for banner
    console.print("\n" * 3)
    
    # Main MOSAICX banner with gradient colors
    say(
        "MOSAICX",
        colors=[MOSAICX_COLORS["secondary"], MOSAICX_COLORS["primary"]],
        align="center",
        font="block",
        space=False,
    )

    # Application name expansion
    expansion = Text(
        "Medical cOmputational Suite for Advanced Intelligent eXtraction",
        style=MOSAICX_COLORS["secondary"],
        justify="center",
    )
    console.print(Align.center(expansion))
    console.print()

    # Institutional attribution
    origin = Text(
        "DIGIT-X Lab @ LMU Radiology | LMU University Hospital", 
        style=MOSAICX_COLORS["primary"]
    )
    console.print(Align.center(origin))
    console.print()

    # Development status capabilities
    capabilities = Text(
        "Intelligent Structuring • Summarization • Classification", 
        style=MOSAICX_COLORS["secondary"]
    )
    console.print(Align.center(capabilities))
    
    # Add spacing after banner
    console.print("\n" * 3)


def styled_message(message: str, style: str = "info", center: bool = False) -> None:
    """
    Display a styled message using predefined color scheme.
    
    Args:
        message (str): The message text to display
        style (str): Style type from MOSAICX_COLORS or Rich style string
        center (bool): Whether to center the message
        
    Example:
        >>> styled_message("Processing complete", "success")
        >>> styled_message("Warning: Check configuration", "warning", center=True)
    """
    # Use predefined colors if available, otherwise use the style directly
    display_style = MOSAICX_COLORS.get(style, style)
    
    if center:
        text = Text(message, style=display_style, justify="center")
        console.print(Align.center(text))
    else:
        console.print(message, style=display_style)


def create_panel(
    content: str, 
    title: Optional[str] = None, 
    style: str = "info"
) -> Panel:
    """
    Create a styled panel for important information display.
    
    Args:
        content (str): Panel content text
        title (str, optional): Panel title
        style (str): Style from MOSAICX_COLORS
        
    Returns:
        Panel: Rich Panel object for display
        
    Example:
        >>> panel = create_panel("System ready", "Status", "success")
        >>> console.print(panel)
    """
    panel_style = MOSAICX_COLORS.get(style, style)
    return Panel(
        content,
        title=title,
        border_style=panel_style,
        title_align="left"
    )


# Initialize banner display when module is imported directly
if __name__ == "__main__":
    show_main_banner()
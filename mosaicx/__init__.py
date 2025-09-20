"""
MOSAICX Package - Medical cOmputational Suite for Advanced Intelligent eXtraction

This package provides comprehensive tools for medical data processing, validation,
and analysis with a focus on intelligent structuring and extraction.

Main Components:
    - mosaicx.display: Terminal interface and banner display
    - mosaicx.mosaicx: Main application entry point and CLI
    - mosaicx.schema: Core schema generation and management modules
    - mosaicx.constants: Centralized configuration and metadata
"""

from .mosaicx import main
from .display import show_main_banner, console

# Import metadata from constants
from .constants import (
    APPLICATION_VERSION as __version__,
    AUTHOR_NAME as __author__,
    AUTHOR_EMAIL as __email__
)

__all__ = ["main", "show_main_banner", "console"]
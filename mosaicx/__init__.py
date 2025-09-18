"""
MOSAICX Package - Medical cOmputational Suite for Advanced Intelligent eXtraction

This package provides comprehensive tools for medical data processing, validation,
and analysis with a focus on intelligent structuring and extraction.

Main Components:
    - mosaicx.display: Terminal interface and banner display
    - mosaicx.mosaicx: Main application entry point
    - mosaicx.schema: Schema management and Pydantic model registry
"""

from .mosaicx import main
from .display import show_main_banner, console

__version__ = "1.0.0"
__author__ = "Lalith Kumar Shiyam Sundar, PhD"
__email__ = "Lalith.shiyam@med.uni-muenchen.de"

__all__ = ["main", "show_main_banner", "console"]
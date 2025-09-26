"""
MOSAICX Package - Medical cOmputational Suite for Advanced Intelligent eXtraction

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Structure first. Insight follows.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab
Department: Department of Radiology
University: LMU University Hospital | LMU Munich

Overview:
---------
Provide cohesive tooling for schema generation, PDF extraction, and report
summarisation, backed by consistent branding and configuration.  Importing the
package exposes the primary console helpers as well as programmatic APIs for
embedding MOSAICX capabilities within larger systems.

Key Modules:
------------
- ``mosaicx.display``: Terminal interface components and banner rendering.
- ``mosaicx.cli`` / ``mosaicx.mosaicx``: Command-line integration with Click.
- ``mosaicx.schema``: Generation pipeline, registry, and stored artifacts.
- ``mosaicx.constants``: Centralised configuration, metadata, and styling.
"""

from .mosaicx import main
from .display import show_main_banner, console
from .api import (
    generate_schema,
    extract_pdf,
    summarize_reports,
    GeneratedSchema,
    ExtractionResult,
)

# Import metadata from constants
from .constants import (
    APPLICATION_VERSION as __version__,
    AUTHOR_NAME as __author__,
    AUTHOR_EMAIL as __email__
)

__all__ = [
    "main",
    "show_main_banner",
    "console",
    "generate_schema",
    "extract_pdf",
    "summarize_reports",
    "GeneratedSchema",
    "ExtractionResult",
]

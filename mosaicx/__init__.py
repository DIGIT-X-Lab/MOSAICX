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

# v2-rewrite: gracefully degrade v1 imports that may not yet work in v2 env
try:
    from .mosaicx import main
    from .display import show_main_banner, console
    from .api import (
        generate_schema,
        extract_pdf,
        summarize_reports,
        GeneratedSchema,
        ExtractionResult,
    )
except Exception:  # noqa: BLE001
    main = None  # type: ignore[assignment]
    show_main_banner = None  # type: ignore[assignment]
    console = None  # type: ignore[assignment]
    generate_schema = None  # type: ignore[assignment]
    extract_pdf = None  # type: ignore[assignment]
    summarize_reports = None  # type: ignore[assignment]
    GeneratedSchema = None  # type: ignore[assignment]
    ExtractionResult = None  # type: ignore[assignment]

# Import metadata from constants
try:
    from .constants import (
        APPLICATION_VERSION as __version__,
        AUTHOR_NAME as __author__,
        AUTHOR_EMAIL as __email__
    )
except Exception:  # noqa: BLE001
    __version__ = "2.0.0a1"
    __author__ = "Lalith Kumar Shiyam Sundar"
    __email__ = "lalith.shiyam@med.uni-muenchen.de"

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

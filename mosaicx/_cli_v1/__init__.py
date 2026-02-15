"""
MOSAICX CLI Package - Console Experience Coordination

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
Expose the rich Click command group that powers the ``mosaicx`` executable
while keeping import-time side effects to a minimum.  This module provides the
publicly supported functions for generating schemas, extracting data, and
producing summaries directly from Python or via console scripts.

Key Exports:
------------
- ``cli``: Root Click group configured with MOSAICX branding.
- ``generate`` / ``extract`` / ``summarize``: Subcommands mapped to API calls.
- ``list_schemas_cmd``: Registry inspection helper for previously generated
  schemas.
- ``main``: Convenience wrapper for invoking the CLI programmatically.
"""

from .app import cli, generate, list_schemas_cmd, extract, summarize, main

__all__ = [
    "cli",
    "generate",
    "list_schemas_cmd",
    "extract",
    "summarize",
    "main",
]

"""
MOSAICX CLI Aggregator - Public Command Entry Points

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
Expose the fully wired Click command group so that both ``python -m mosaicx``
and installed console entry points delegate to a single, well documented
surface.  Importing this module brings the CLI helpers into the package-level
namespace without triggering side effects such as banner rendering.

Core Responsibilities:
----------------------
- Bridge ``mosaicx.cli`` helpers into ``mosaicx`` for ``__all__`` exports.
- Provide a conventional ``main()`` for setuptools-style console scripts.
- Preserve a guard for module execution so ``python -m mosaicx.mosaicx`` works.
"""

from __future__ import annotations

from .cli import cli, extract, generate, list_schemas_cmd, main, summarize

__all__ = [
    "cli",
    "generate",
    "list_schemas_cmd",
    "extract",
    "summarize",
    "main",
]


if __name__ == "__main__":
    main()

"""
MOSAICX Module Entrypoint - ``python -m mosaicx`` Launcher

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
Permit the package to be executed directly via ``python -m mosaicx``.  This
wrapper imports the CLI group without rendering banners and delegates execution
to the same command graph exposed by the console script.
"""

from mosaicx.mosaicx import cli

if __name__ == "__main__":
    cli()

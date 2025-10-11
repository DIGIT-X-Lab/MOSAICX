"""
Bundled schema templates shipped with MOSAICX.

These schemas provide ready-to-use starting points for common extraction
scenarios. End users can copy or extend them in their own workspace by placing
additional `.py` files under ``~/.mosaicx/schemas`` (see :mod:`mosaicx.constants`).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["TEMPLATE_ROOT", "PYTHON_TEMPLATE_ROOT", "JSON_TEMPLATE_ROOT"]

TEMPLATE_ROOT = Path(__file__).resolve().parent
PYTHON_TEMPLATE_ROOT = TEMPLATE_ROOT / "python"
JSON_TEMPLATE_ROOT = TEMPLATE_ROOT / "json"

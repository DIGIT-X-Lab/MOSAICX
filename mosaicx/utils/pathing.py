"""
MOSAICX Path Utilities - Schema Asset Resolution

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
Provide resilient filesystem lookups for generated schema modules regardless
of how users reference them (registry ID, filename, or explicit path).  The
helpers favour MOSAICX's managed directories while still respecting manual
paths and the current working directory to support scripting workflows.

Key Behaviours:
--------------
- Inspect the schema registry for canonical locations before falling back to
  direct path resolution.
- Normalise legacy references by automatically appending ``.py`` when absent.
- Guard against missing assets by returning ``None`` rather than raising,
  allowing caller-controlled error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..constants import PACKAGE_SCHEMA_PYD_DIR
from ..schema.registry import get_schema_by_id


def resolve_schema_reference(schema_ref: str) -> Optional[Path]:
    """Resolve a schema identifier (ID, filename, or path) to a filesystem path."""

    schema_by_id = get_schema_by_id(schema_ref)
    if schema_by_id:
        schema_path = Path(schema_by_id["file_path"])
        if schema_path.exists():
            return schema_path

    schema_dir = Path(PACKAGE_SCHEMA_PYD_DIR)
    if schema_dir.exists():
        direct = schema_dir / schema_ref
        if direct.exists() and direct.suffix == ".py":
            return direct

        if not schema_ref.endswith(".py"):
            with_ext = schema_dir / f"{schema_ref}.py"
            if with_ext.exists():
                return with_ext

    explicit = Path(schema_ref)
    if explicit.exists() and explicit.suffix == ".py":
        return explicit

    if not explicit.is_absolute():
        relative = Path.cwd() / schema_ref
        if relative.exists() and relative.suffix == ".py":
            return relative

    return None


__all__ = ["resolve_schema_reference"]

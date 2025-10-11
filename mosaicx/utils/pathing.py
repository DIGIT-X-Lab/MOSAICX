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
from typing import Iterable, Optional

from ..constants import (
    PACKAGE_SCHEMA_TEMPLATES_PY_DIR,
    USER_SCHEMA_DIR,
)
from ..schema.registry import get_schema_by_id


def resolve_schema_reference(schema_ref: str) -> Optional[Path]:
    """Resolve a schema identifier (ID, filename, or path) to a filesystem path."""

    schema_by_id = get_schema_by_id(schema_ref)
    fallback_name: Optional[str] = None
    if schema_by_id:
        schema_path = Path(schema_by_id["file_path"])
        if schema_path.exists():
            return schema_path
        fallback_name = schema_by_id.get("file_name")

    search_roots: Iterable[Path] = (
        root
        for root in [USER_SCHEMA_DIR, PACKAGE_SCHEMA_TEMPLATES_PY_DIR]
        if isinstance(root, Path)
    )
    for root in search_roots:
        if not root.exists():
            continue

        for candidate_name in (schema_ref, fallback_name):
            if not candidate_name:
                continue
            candidate_path = root / candidate_name
            if candidate_path.exists() and candidate_path.suffix == ".py":
                return candidate_path

            if not candidate_name.endswith(".py"):
                with_ext = root / f"{candidate_name}.py"
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

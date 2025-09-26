"""Path and lookup helpers for schema assets."""

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

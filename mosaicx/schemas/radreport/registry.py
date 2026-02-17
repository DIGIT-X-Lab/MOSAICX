"""RadReport template registry — discovery via YAML file scanning.

Built-in templates live as YAML files in ``templates/`` alongside this
module.  This registry scans those files on first access and exposes
them through the same ``list_templates()`` / ``get_template()`` API
that the rest of the codebase expects.

The YAML files are the single source of truth — adding a new built-in
template requires only a new ``.yaml`` file, no Python changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TemplateInfo:
    """Metadata for a registered RadReport template."""

    name: str
    exam_type: str
    radreport_id: Optional[str] = None
    description: str = ""
    mode: Optional[str] = None  # pipeline mode for auto-detection


# ---------------------------------------------------------------------------
# YAML scanning
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_cache: list[TemplateInfo] | None = None
_cache_map: dict[str, TemplateInfo] | None = None


def _scan_yaml_templates() -> list[TemplateInfo]:
    """Scan the built-in ``templates/`` directory and return TemplateInfo list."""
    import yaml

    templates: list[TemplateInfo] = []
    if not _TEMPLATES_DIR.is_dir():
        return templates

    for path in sorted(_TEMPLATES_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        stem = path.stem
        templates.append(
            TemplateInfo(
                name=stem,
                exam_type=stem,
                radreport_id=data.get("radreport_id"),
                description=data.get("description", ""),
                mode=data.get("mode"),
            )
        )

    return templates


def _ensure_loaded() -> None:
    """Populate the cache on first access."""
    global _cache, _cache_map
    if _cache is None:
        _cache = _scan_yaml_templates()
        _cache_map = {t.name: t for t in _cache}


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------


def list_templates() -> list[TemplateInfo]:
    """Return all registered templates."""
    _ensure_loaded()
    assert _cache is not None
    return list(_cache)


def get_template(name: str) -> Optional[TemplateInfo]:
    """Get a template by name. Returns None if not found."""
    _ensure_loaded()
    assert _cache_map is not None
    return _cache_map.get(name)

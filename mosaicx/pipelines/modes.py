"""Extraction mode registry.

Modes are specialized multi-step pipelines for specific document domains.
Each mode is a DSPy Module registered via the ``@register_mode`` decorator.
"""
from __future__ import annotations

from typing import Any

MODES: dict[str, type] = {}


def register_mode(name: str, description: str):
    """Decorator to register an extraction mode."""
    def decorator(cls):
        cls.mode_name = name
        cls.mode_description = description
        MODES[name] = cls
        return cls
    return decorator


def get_mode(name: str) -> type:
    """Get a registered mode by name. Raises ValueError if not found."""
    if name not in MODES:
        available = ", ".join(sorted(MODES)) or "(none)"
        raise ValueError(f"Unknown mode: {name!r}. Available: {available}")
    return MODES[name]


def list_modes() -> list[tuple[str, str]]:
    """Return list of (name, description) tuples for all registered modes."""
    return [(name, cls.mode_description) for name, cls in sorted(MODES.items())]

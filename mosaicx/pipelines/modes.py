"""Extraction mode registry.

Modes are specialized multi-step pipelines for specific document domains.
Each mode is a DSPy Module registered via the ``@register_mode`` decorator.

Eager metadata registration is supported via ``register_mode_info`` so that
``list_modes`` works without importing DSPy.
"""
from __future__ import annotations

from typing import Any

MODES: dict[str, type] = {}
_MODE_DESCRIPTIONS: dict[str, str] = {}


def register_mode_info(name: str, description: str) -> None:
    """Eagerly register mode metadata (name + description) without a class.

    This allows ``list_modes()`` to return the mode even before the DSPy
    class has been lazily loaded.
    """
    _MODE_DESCRIPTIONS[name] = description


def register_mode(name: str, description: str):
    """Decorator to register an extraction mode."""
    def decorator(cls):
        cls.mode_name = name
        cls.mode_description = description
        MODES[name] = cls
        _MODE_DESCRIPTIONS[name] = description
        return cls
    return decorator


def get_mode(name: str) -> type:
    """Get a registered mode by name. Raises ValueError if not found."""
    if name not in MODES:
        available = ", ".join(sorted(_MODE_DESCRIPTIONS)) or "(none)"
        raise ValueError(f"Unknown mode: {name!r}. Available: {available}")
    return MODES[name]


def list_modes() -> list[tuple[str, str]]:
    """Return list of (name, description) tuples for all registered modes."""
    return [(name, _MODE_DESCRIPTIONS[name]) for name in sorted(_MODE_DESCRIPTIONS)]

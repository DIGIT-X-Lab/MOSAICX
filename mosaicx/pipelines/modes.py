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
        # Mode metadata was eagerly registered but class not yet loaded.
        # Trigger lazy loading by accessing the DSPy class from the module.
        if name in _MODE_DESCRIPTIONS:
            _trigger_lazy_load(name)
        if name not in MODES:
            available = ", ".join(sorted(_MODE_DESCRIPTIONS)) or "(none)"
            raise ValueError(f"Unknown mode: {name!r}. Available: {available}")
    return MODES[name]


# Map of mode name → module that provides it (for lazy loading)
_MODE_MODULES: dict[str, str] = {
    "radiology": "mosaicx.pipelines.radiology",
    "pathology": "mosaicx.pipelines.pathology",
}


def _trigger_lazy_load(name: str) -> None:
    """Force the DSPy class to be built for a given mode name."""
    mod_name = _MODE_MODULES.get(name)
    if mod_name is None:
        return
    import importlib
    mod = importlib.import_module(mod_name)
    # Accessing the main class triggers module __getattr__ → _build_dspy_classes()
    _LAZY_CLASS_NAMES = {
        "radiology": "RadiologyReportStructurer",
        "pathology": "PathologyReportStructurer",
    }
    cls_name = _LAZY_CLASS_NAMES.get(name)
    if cls_name:
        getattr(mod, cls_name, None)


def list_modes() -> list[tuple[str, str]]:
    """Return list of (name, description) tuples for all registered modes."""
    return [(name, _MODE_DESCRIPTIONS[name]) for name in sorted(_MODE_DESCRIPTIONS)]

"""Conformance registry for de-identification standards.

Each conformance defines the PHI categories, regex patterns, and LLM prompt
fragment for a specific privacy standard (e.g., HIPAA, GDPR).

External packages register conformances by calling ``register_conformance()``
at import time. See ``mosaicx/conformance/README.md`` for the plugin interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConformanceSpec:
    """Definition of a de-identification conformance standard."""

    name: str
    description: str
    phi_categories: list[str]
    regex_patterns: list[tuple[re.Pattern, str]]
    prompt_fragment: str


_REGISTRY: dict[str, ConformanceSpec] = {}


def register_conformance(spec: ConformanceSpec) -> None:
    """Register a conformance mode. Called at import time by each standard."""
    _REGISTRY[spec.name] = spec


def get_conformance(name: str) -> ConformanceSpec:
    """Look up a registered conformance by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(
            f"Unknown conformance '{name}'. Available: {available}. "
            f"Install additional packages for more conformance modes."
        )
    return _REGISTRY[name]


def list_conformances() -> list[str]:
    """Return names of all registered conformances."""
    return sorted(_REGISTRY)

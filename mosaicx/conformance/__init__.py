"""Conformance standards for de-identification.

Importing this package auto-registers built-in conformances (HIPAA).
External packages can register additional conformances by calling
``register_conformance()`` with a ``ConformanceSpec``.
"""
from __future__ import annotations

from .registry import ConformanceSpec, get_conformance, list_conformances, register_conformance

__all__ = [
    "ConformanceSpec",
    "get_conformance",
    "list_conformances",
    "register_conformance",
]

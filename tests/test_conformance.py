"""Tests for the conformance registry."""
from __future__ import annotations

import re

import pytest


def test_register_and_get_conformance():
    from mosaicx.conformance.registry import ConformanceSpec, get_conformance, register_conformance

    spec = ConformanceSpec(
        name="test_standard",
        description="Test standard",
        phi_categories=["NAME", "DATE"],
        regex_patterns=[
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
        ],
        prompt_fragment="Detect names and dates.",
    )
    register_conformance(spec)
    result = get_conformance("test_standard")
    assert result.name == "test_standard"
    assert result.phi_categories == ["NAME", "DATE"]
    assert len(result.regex_patterns) == 1


def test_get_unknown_conformance_raises():
    from mosaicx.conformance.registry import get_conformance

    with pytest.raises(KeyError, match="Unknown conformance"):
        get_conformance("nonexistent_standard_xyz")


def test_list_conformances():
    from mosaicx.conformance.registry import list_conformances

    result = list_conformances()
    assert isinstance(result, list)

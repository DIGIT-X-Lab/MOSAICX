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


def test_hipaa_registered_on_import():
    import mosaicx.conformance  # noqa: F401
    from mosaicx.conformance.registry import get_conformance

    spec = get_conformance("hipaa")
    assert spec.name == "hipaa"
    assert "NAME" in spec.phi_categories
    assert "DATE" in spec.phi_categories
    assert "SSN" in spec.phi_categories
    assert "MRN" in spec.phi_categories
    assert len(spec.regex_patterns) > 0
    assert len(spec.prompt_fragment) > 0


def test_hipaa_regex_catches_ssn():
    import mosaicx.conformance  # noqa: F401
    from mosaicx.conformance.registry import get_conformance

    spec = get_conformance("hipaa")
    text = "SSN: 123-45-6789"
    matches = []
    for pattern, phi_type in spec.regex_patterns:
        for m in pattern.finditer(text):
            matches.append((m.group(), phi_type))
    assert any(phi_type == "SSN" for _, phi_type in matches)


def test_hipaa_regex_no_false_positive_on_clean_text():
    import mosaicx.conformance  # noqa: F401
    from mosaicx.conformance.registry import get_conformance

    spec = get_conformance("hipaa")
    text = "The lungs are clear. No pleural effusion."
    matches = []
    for pattern, phi_type in spec.regex_patterns:
        for m in pattern.finditer(text):
            matches.append(m.group())
    assert len(matches) == 0


def test_hipaa_in_list_conformances():
    import mosaicx.conformance  # noqa: F401
    from mosaicx.conformance.registry import list_conformances

    assert "hipaa" in list_conformances()

# tests/test_public_api.py
"""Tests for the mosaicx top-level public API functions.

Verifies that the four convenience functions (extract, summarize,
generate_schema, deidentify) are importable, callable, and have the
expected signatures.  Also tests the regex-only deidentify path which
requires no LLM.
"""

import inspect

import pytest


class TestPublicAPI:
    """Verify the public API functions are importable and callable."""

    def test_extract_function(self):
        from mosaicx import extract

        assert callable(extract)

    def test_summarize_function(self):
        from mosaicx import summarize

        assert callable(summarize)

    def test_generate_schema_function(self):
        from mosaicx import generate_schema

        assert callable(generate_schema)

    def test_deidentify_function(self):
        from mosaicx import deidentify

        assert callable(deidentify)

    def test_all_exports(self):
        """All four functions should appear in __all__."""
        import mosaicx

        for name in ("extract", "summarize", "generate_schema", "deidentify"):
            assert name in mosaicx.__all__, f"{name!r} missing from __all__"


class TestFunctionSignatures:
    """Verify each wrapper has the documented parameters."""

    def test_extract_signature(self):
        from mosaicx import extract

        sig = inspect.signature(extract)
        params = list(sig.parameters.keys())
        assert "document_path" in params
        assert "schema" in params
        assert "mode" in params
        assert "template" in params
        # All three should default to None
        assert sig.parameters["schema"].default is None
        assert sig.parameters["mode"].default is None
        assert sig.parameters["template"].default is None

    def test_summarize_signature(self):
        from mosaicx import summarize

        sig = inspect.signature(summarize)
        params = list(sig.parameters.keys())
        assert "document_paths" in params

    def test_generate_schema_signature(self):
        from mosaicx import generate_schema

        sig = inspect.signature(generate_schema)
        params = list(sig.parameters.keys())
        assert "description" in params
        assert "example_text" in params
        assert sig.parameters["example_text"].default is None

    def test_deidentify_signature(self):
        from mosaicx import deidentify

        sig = inspect.signature(deidentify)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert "mode" in params
        assert sig.parameters["mode"].default == "remove"


class TestDeidentifyRegex:
    """Test the regex-only path of deidentify (no LLM needed)."""

    def test_ssn_scrubbed(self):
        from mosaicx import deidentify

        result = deidentify("Patient SSN 123-45-6789", mode="regex")
        assert "123-45-6789" not in result
        assert "[REDACTED]" in result

    def test_phone_scrubbed(self):
        from mosaicx import deidentify

        result = deidentify("Call 555-123-4567 for info", mode="regex")
        assert "555-123-4567" not in result
        assert "[REDACTED]" in result

    def test_email_scrubbed(self):
        from mosaicx import deidentify

        result = deidentify("Contact: john.doe@hospital.com", mode="regex")
        assert "john.doe@hospital.com" not in result
        assert "[REDACTED]" in result

    def test_mrn_scrubbed(self):
        from mosaicx import deidentify

        result = deidentify("MRN: 12345678", mode="regex")
        assert "12345678" not in result
        assert "[REDACTED]" in result

    def test_no_phi_unchanged(self):
        from mosaicx import deidentify

        clean = "Normal chest radiograph. No acute findings."
        result = deidentify(clean, mode="regex")
        assert result == clean

    def test_regex_returns_string(self):
        from mosaicx import deidentify

        result = deidentify("Some text 123-45-6789", mode="regex")
        assert isinstance(result, str)

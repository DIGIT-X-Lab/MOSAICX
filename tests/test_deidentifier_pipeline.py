# tests/test_deidentifier_pipeline.py
"""Tests for the de-identification pipeline."""

import pytest


class TestPHIRegex:
    """Test the deterministic regex guard patterns."""

    def test_ssn_detection(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "SSN: 123-45-6789"
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert any(m for m in matches if m)

    def test_phone_detection(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "Phone: (555) 123-4567"
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert any(m for m in matches if m)

    def test_mrn_detection(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "MRN: 12345678"
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert any(m for m in matches if m)

    def test_email_detection(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "Contact: john.doe@hospital.com"
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert any(m for m in matches if m)

    def test_clean_text_no_match(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "The lungs are clear. No pleural effusion."
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert not any(m for m in matches if m)


class TestRegexScrub:
    def test_scrubs_ssn(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi
        text = "Patient SSN 123-45-6789 presents with cough."
        scrubbed = regex_scrub_phi(text)
        assert "123-45-6789" not in scrubbed
        assert "cough" in scrubbed

    def test_scrubs_phone(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi
        text = "Call (555) 123-4567 for results."
        scrubbed = regex_scrub_phi(text)
        assert "(555) 123-4567" not in scrubbed

    def test_clean_text_unchanged(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi
        text = "Normal chest radiograph. No acute findings."
        scrubbed = regex_scrub_phi(text)
        assert scrubbed == text


class TestDeidentifierSignature:
    def test_redact_phi_signature(self):
        from mosaicx.pipelines.deidentifier import RedactPHI
        assert "document_text" in RedactPHI.input_fields
        assert "redacted_text" in RedactPHI.output_fields


class TestDeidentifierModule:
    def test_module_has_submodules(self):
        from mosaicx.pipelines.deidentifier import Deidentifier
        d = Deidentifier()
        assert hasattr(d, "redact")

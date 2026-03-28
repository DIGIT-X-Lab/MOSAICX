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


# ---------------------------------------------------------------------------
# PHI_PATTERN_TYPES alignment
# ---------------------------------------------------------------------------


class TestPHIPatternTypes:
    """Verify that PHI_PATTERN_TYPES is aligned with PHI_PATTERNS."""

    def test_same_length(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS, PHI_PATTERN_TYPES
        assert len(PHI_PATTERNS) == len(PHI_PATTERN_TYPES)

    def test_all_types_are_strings(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERN_TYPES
        for t in PHI_PATTERN_TYPES:
            assert isinstance(t, str) and len(t) > 0


# ---------------------------------------------------------------------------
# regex_scrub_phi_with_mappings
# ---------------------------------------------------------------------------


class TestRegexScrubWithMappings:
    """Test the enhanced regex scrubber that returns redaction mappings."""

    def test_ssn_mapping(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "Patient SSN 123-45-6789 presents with cough."
        scrubbed, mappings = regex_scrub_phi_with_mappings(text)
        assert "123-45-6789" not in scrubbed
        assert len(mappings) >= 1
        ssn_map = [m for m in mappings if m["phi_type"] == "SSN"]
        assert len(ssn_map) == 1
        assert ssn_map[0]["original"] == "123-45-6789"
        assert ssn_map[0]["method"] == "regex"
        assert ssn_map[0]["replacement"] == "[REDACTED]"
        assert ssn_map[0]["start"] == text.index("123-45-6789")
        assert ssn_map[0]["end"] == text.index("123-45-6789") + len("123-45-6789")

    def test_phone_mapping(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "Call (555) 123-4567 for results."
        scrubbed, mappings = regex_scrub_phi_with_mappings(text)
        assert "(555) 123-4567" not in scrubbed
        phone_map = [m for m in mappings if m["phi_type"] == "PHONE"]
        assert len(phone_map) >= 1
        assert phone_map[0]["method"] == "regex"

    def test_email_mapping(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "Contact dr.smith@hospital.com for info."
        scrubbed, mappings = regex_scrub_phi_with_mappings(text)
        assert "dr.smith@hospital.com" not in scrubbed
        email_map = [m for m in mappings if m["phi_type"] == "EMAIL"]
        assert len(email_map) == 1
        assert email_map[0]["original"] == "dr.smith@hospital.com"

    def test_mrn_mapping(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "MRN: 12345678 admitted today."
        scrubbed, mappings = regex_scrub_phi_with_mappings(text)
        assert "12345678" not in scrubbed
        mrn_map = [m for m in mappings if m["phi_type"] == "MRN"]
        assert len(mrn_map) == 1
        assert mrn_map[0]["method"] == "regex"

    def test_date_mapping(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "DOB: 01/15/1980 seen on 2026-03-28."
        scrubbed, mappings = regex_scrub_phi_with_mappings(text)
        assert "01/15/1980" not in scrubbed
        assert "2026-03-28" not in scrubbed
        date_maps = [m for m in mappings if m["phi_type"] == "DATE"]
        assert len(date_maps) == 2

    def test_multiple_phi_types(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "John Doe SSN 123-45-6789 email john@hosp.com DOB 01/01/1990."
        scrubbed, mappings = regex_scrub_phi_with_mappings(text)
        types_found = {m["phi_type"] for m in mappings}
        assert "SSN" in types_found
        assert "EMAIL" in types_found
        assert "DATE" in types_found

    def test_clean_text_no_mappings(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "Normal chest radiograph. No acute findings."
        scrubbed, mappings = regex_scrub_phi_with_mappings(text)
        assert scrubbed == text
        assert mappings == []

    def test_positions_are_in_original(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "SSN 123-45-6789 and MRN: 99887766 here."
        scrubbed, mappings = regex_scrub_phi_with_mappings(text)
        for m in mappings:
            assert text[m["start"]:m["end"]] == m["original"]

    def test_scrubbed_text_matches_original_function(self):
        """The scrubbed text from regex_scrub_phi_with_mappings should match
        what regex_scrub_phi produces for the same input."""
        from mosaicx.pipelines.deidentifier import (
            regex_scrub_phi,
            regex_scrub_phi_with_mappings,
        )
        text = "SSN 123-45-6789 phone (555) 444-3333 email a@b.com date 01/01/2025."
        expected = regex_scrub_phi(text)
        actual, _ = regex_scrub_phi_with_mappings(text)
        assert actual == expected

    def test_mappings_sorted_by_start(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi_with_mappings
        text = "SSN 123-45-6789 email a@b.com phone (555) 444-3333."
        _, mappings = regex_scrub_phi_with_mappings(text)
        starts = [m["start"] for m in mappings]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# _compute_redaction_mappings
# ---------------------------------------------------------------------------


class TestComputeRedactionMappings:
    """Test the diff-based redaction mapping function."""

    def test_single_redaction(self):
        from mosaicx.pipelines.deidentifier import _compute_redaction_mappings
        original = "Patient John Doe has pneumonia."
        redacted = "Patient [REDACTED] has pneumonia."
        mappings = _compute_redaction_mappings(original, redacted)
        assert len(mappings) == 1
        assert mappings[0]["original"] == "John Doe"
        assert mappings[0]["start"] == 8
        assert mappings[0]["end"] == 16
        assert mappings[0]["replacement"] == "[REDACTED]"

    def test_multiple_redactions(self):
        from mosaicx.pipelines.deidentifier import _compute_redaction_mappings
        original = "John Doe SSN 123-45-6789 at City Hospital."
        redacted = "[REDACTED] SSN [REDACTED] at [REDACTED]."
        mappings = _compute_redaction_mappings(original, redacted)
        assert len(mappings) == 3
        assert mappings[0]["original"] == "John Doe"
        assert mappings[1]["original"] == "123-45-6789"
        assert mappings[2]["original"] == "City Hospital"

    def test_no_redactions(self):
        from mosaicx.pipelines.deidentifier import _compute_redaction_mappings
        text = "No PHI here."
        mappings = _compute_redaction_mappings(text, text)
        assert mappings == []

    def test_redaction_at_end(self):
        from mosaicx.pipelines.deidentifier import _compute_redaction_mappings
        original = "Contact John Doe"
        redacted = "Contact [REDACTED]"
        mappings = _compute_redaction_mappings(original, redacted)
        assert len(mappings) == 1
        assert mappings[0]["original"] == "John Doe"

    def test_redaction_at_start(self):
        from mosaicx.pipelines.deidentifier import _compute_redaction_mappings
        original = "John Doe has cough."
        redacted = "[REDACTED] has cough."
        mappings = _compute_redaction_mappings(original, redacted)
        assert len(mappings) == 1
        assert mappings[0]["original"] == "John Doe"
        assert mappings[0]["start"] == 0

    def test_positions_reference_original(self):
        from mosaicx.pipelines.deidentifier import _compute_redaction_mappings
        original = "Dr. Jane Smith at 123 Main St treated MRN: 99887766."
        redacted = "Dr. [REDACTED] at [REDACTED] treated MRN: [REDACTED]."
        mappings = _compute_redaction_mappings(original, redacted)
        for m in mappings:
            assert original[m["start"]:m["end"]] == m["original"]


# ---------------------------------------------------------------------------
# _label_phi_types
# ---------------------------------------------------------------------------


class TestLabelPHITypes:
    """Test PHI type labeling for redaction mappings."""

    def test_ssn_labeled(self):
        from mosaicx.pipelines.deidentifier import _label_phi_types
        mappings = [{"original": "123-45-6789", "start": 0, "end": 11}]
        result = _label_phi_types("123-45-6789", mappings)
        assert result[0]["phi_type"] == "SSN"
        assert result[0]["method"] == "regex"

    def test_email_labeled(self):
        from mosaicx.pipelines.deidentifier import _label_phi_types
        mappings = [{"original": "john@hospital.com", "start": 0, "end": 17}]
        result = _label_phi_types("john@hospital.com", mappings)
        assert result[0]["phi_type"] == "EMAIL"
        assert result[0]["method"] == "regex"

    def test_name_labeled_other(self):
        from mosaicx.pipelines.deidentifier import _label_phi_types
        mappings = [{"original": "John Doe", "start": 0, "end": 8}]
        result = _label_phi_types("John Doe", mappings)
        assert result[0]["phi_type"] == "OTHER"
        assert result[0]["method"] == "llm"

    def test_date_labeled(self):
        from mosaicx.pipelines.deidentifier import _label_phi_types
        mappings = [{"original": "01/15/1980", "start": 0, "end": 10}]
        result = _label_phi_types("01/15/1980", mappings)
        assert result[0]["phi_type"] == "DATE"
        assert result[0]["method"] == "regex"

    def test_mixed_types(self):
        from mosaicx.pipelines.deidentifier import _label_phi_types
        mappings = [
            {"original": "John Doe", "start": 0, "end": 8},
            {"original": "123-45-6789", "start": 13, "end": 24},
            {"original": "john@hospital.com", "start": 30, "end": 47},
        ]
        original = "John Doe SSN 123-45-6789 email john@hospital.com"
        result = _label_phi_types(original, mappings)
        assert result[0]["phi_type"] == "OTHER"
        assert result[0]["method"] == "llm"
        assert result[1]["phi_type"] == "SSN"
        assert result[1]["method"] == "regex"
        assert result[2]["phi_type"] == "EMAIL"
        assert result[2]["method"] == "regex"


# ---------------------------------------------------------------------------
# _merge_mappings
# ---------------------------------------------------------------------------


class TestMergeMappings:
    """Test merging of diff-based and regex-based mappings."""

    def test_no_overlap(self):
        from mosaicx.pipelines.deidentifier import _merge_mappings
        diff = [{"original": "John Doe", "start": 0, "end": 8,
                 "phi_type": "OTHER", "method": "llm",
                 "replacement": "[REDACTED]"}]
        regex = [{"original": "123-45-6789", "start": 13, "end": 24,
                  "phi_type": "SSN", "method": "regex",
                  "replacement": "[REDACTED]"}]
        merged = _merge_mappings(diff, regex)
        assert len(merged) == 2
        # Sorted by start
        assert merged[0]["original"] == "John Doe"
        assert merged[1]["original"] == "123-45-6789"

    def test_overlap_deduplicates(self):
        from mosaicx.pipelines.deidentifier import _merge_mappings
        # Same SSN detected by both diff and regex
        diff = [{"original": "123-45-6789", "start": 4, "end": 15,
                 "phi_type": "SSN", "method": "regex",
                 "replacement": "[REDACTED]"}]
        regex = [{"original": "123-45-6789", "start": 4, "end": 15,
                  "phi_type": "SSN", "method": "regex",
                  "replacement": "[REDACTED]"}]
        merged = _merge_mappings(diff, regex)
        # Should only have 1 entry (regex wins, diff is deduplicated)
        assert len(merged) == 1
        assert merged[0]["phi_type"] == "SSN"

    def test_partial_overlap(self):
        from mosaicx.pipelines.deidentifier import _merge_mappings
        # Diff caught a larger span that overlaps with regex
        diff = [{"original": "SSN 123-45-6789", "start": 0, "end": 15,
                 "phi_type": "SSN", "method": "regex",
                 "replacement": "[REDACTED]"}]
        regex = [{"original": "123-45-6789", "start": 4, "end": 15,
                  "phi_type": "SSN", "method": "regex",
                  "replacement": "[REDACTED]"}]
        merged = _merge_mappings(diff, regex)
        # Diff overlaps with regex, so only regex kept
        assert len(merged) == 1
        assert merged[0]["start"] == 4

    def test_empty_inputs(self):
        from mosaicx.pipelines.deidentifier import _merge_mappings
        assert _merge_mappings([], []) == []
        assert len(_merge_mappings([], [{"original": "x", "start": 0, "end": 1,
                                         "phi_type": "SSN", "method": "regex",
                                         "replacement": "[REDACTED]"}])) == 1

    def test_result_sorted_by_start(self):
        from mosaicx.pipelines.deidentifier import _merge_mappings
        diff = [{"original": "John", "start": 20, "end": 24,
                 "phi_type": "OTHER", "method": "llm",
                 "replacement": "[REDACTED]"}]
        regex = [{"original": "123-45-6789", "start": 5, "end": 16,
                  "phi_type": "SSN", "method": "regex",
                  "replacement": "[REDACTED]"}]
        merged = _merge_mappings(diff, regex)
        starts = [m["start"] for m in merged]
        assert starts == sorted(starts)


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

# tests/test_source_mapping.py
"""Tests for the unified _source block builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaicx.source_mapping import (
    _date_alternatives,
    _find_tight_excerpt,
    _flatten_dict,
    build_source_block,
)


class TestFlattenDict:
    def test_flat_dict(self):
        assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        assert _flatten_dict({"a": {"b": 1}}) == {"a.b": 1}

    def test_list_of_objects(self):
        result = _flatten_dict({"items": [{"x": 1}, {"x": 2}]})
        assert result == {"items[0].x": 1, "items[1].x": 2}

    def test_scalar_list(self):
        result = _flatten_dict({"tags": ["a", "b"]})
        assert result == {"tags[0]": "a", "tags[1]": "b"}

    def test_deep_nesting(self):
        result = _flatten_dict({"a": {"b": [{"c": {"d": 1}}]}})
        assert result == {"a.b[0].c.d": 1}

    def test_none_value(self):
        result = _flatten_dict({"a": None})
        assert result == {"a": None}

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}

    def test_mixed(self):
        result = _flatten_dict({
            "name": "Sarah",
            "findings": [
                {"location": "right lobe", "size": 8.5},
                {"location": "left hilum", "size": 12.0},
            ],
        })
        assert result == {
            "name": "Sarah",
            "findings[0].location": "right lobe",
            "findings[0].size": 8.5,
            "findings[1].location": "left hilum",
            "findings[1].size": 12.0,
        }


class TestDateAlternatives:
    def test_iso_date(self):
        alts = _date_alternatives("1985-03-15")
        assert "March 15, 1985" in alts
        assert "15.03.1985" in alts
        assert "03/15/1985" in alts

    def test_european_date(self):
        alts = _date_alternatives("15.03.1985")
        assert "March 15, 1985" in alts
        assert "1985-03-15" in alts

    def test_invalid_date(self):
        assert _date_alternatives("not-a-date") == []

    def test_empty(self):
        assert _date_alternatives("") == []


class TestFindTightExcerpt:
    def test_exact_match(self):
        excerpt, start, end = _find_tight_excerpt(
            "Patient Name: Sarah Johnson\nPatient ID: PID-12345",
            "Sarah Johnson",
        )
        assert excerpt == "Patient Name: Sarah Johnson"
        assert start == 14
        assert end == 27

    def test_case_insensitive(self):
        excerpt, start, end = _find_tight_excerpt(
            "Gender: FEMALE",
            "female",
        )
        assert excerpt is not None
        assert "FEMALE" in excerpt

    def test_date_format_conversion(self):
        excerpt, start, end = _find_tight_excerpt(
            "Date of Birth: March 15, 1985\nAge: 40",
            "1985-03-15",
        )
        assert excerpt is not None
        assert "March 15, 1985" in excerpt
        assert start is not None

    def test_not_found(self):
        excerpt, start, end = _find_tight_excerpt(
            "Patient Name: Sarah Johnson",
            "NONEXISTENT",
        )
        assert excerpt is None
        assert start is None

    def test_empty_value(self):
        assert _find_tight_excerpt("some text", "") == (None, None, None)

    def test_empty_source(self):
        assert _find_tight_excerpt("", "value") == (None, None, None)

    def test_numeric_value(self):
        excerpt, start, end = _find_tight_excerpt(
            "BMI 23.4 18.5-24.9 Normal",
            "23.4",
        )
        assert excerpt is not None
        assert "23.4" in excerpt


class TestBuildSourceBlock:
    def _make_doc(self, text="Patient Name: Sarah Johnson", fmt="pdf"):
        from mosaicx.documents.models import LoadedDocument, TextBlock

        blocks = []
        # Simple: one text block covering the whole text
        if text.strip():
            blocks.append(TextBlock(
                text=text, start=0, end=len(text), page=1,
                bbox=(0.0, 0.0, 300.0, 20.0),
            ))

        return LoadedDocument(
            text=text,
            source_path=Path("test.pdf"),
            format=fmt,
            page_count=1,
            text_blocks=blocks,
            page_dimensions=[(612.0, 792.0)] if fmt == "pdf" else [],
        )

    def test_extraction_mode(self):
        doc = self._make_doc()
        result = build_source_block(doc, fields={"patient_name": "Sarah Johnson"})

        assert "_guide" in result
        assert "fields" in result
        assert result["_guide"]["version"] == "1.0"
        assert result["_guide"]["coordinate_space"] == "pdf_points"
        assert "patient_name" in result["fields"]
        assert result["fields"]["patient_name"]["grounded"] is True

    def test_deidentification_mode(self):
        doc = self._make_doc()
        rmap = [{"original": "Sarah Johnson", "replacement": "[REDACTED]", "phi_type": "NAME"}]
        result = build_source_block(doc, redaction_map=rmap)

        assert "name_0" in result["fields"]
        field = result["fields"]["name_0"]
        assert field["value"] == "Sarah Johnson"
        assert field["replacement"] == "[REDACTED]"
        assert field["phi_type"] == "NAME"

    def test_text_file_no_coordinates(self):
        doc = self._make_doc(fmt="txt")
        result = build_source_block(doc, fields={"name": "Sarah Johnson"})

        assert result["_guide"]["coordinate_space"] == "none"
        # Excerpt found but no spans (no text_blocks for txt)
        # Actually we added a text_block in _make_doc, but format is txt
        # The coordinate_space is "none" which is what matters

    def test_image_format(self):
        doc = self._make_doc(fmt="png")
        result = build_source_block(doc, fields={"name": "Sarah Johnson"})

        assert result["_guide"]["coordinate_space"] == "image_pixels"
        assert result["_guide"]["origin"] == "top-left"
        assert "to_fitz_rect" not in result["_guide"]

    def test_nested_fields_flattened(self):
        doc = self._make_doc(text="right lobe 8.5mm left hilum 12mm")
        result = build_source_block(doc, fields={
            "findings": [
                {"location": "right lobe", "size": "8.5mm"},
                {"location": "left hilum", "size": "12mm"},
            ],
        })
        assert "findings[0].location" in result["fields"]
        assert "findings[1].location" in result["fields"]

    def test_guide_has_page_dimensions(self):
        doc = self._make_doc()
        result = build_source_block(doc, fields={"name": "Sarah Johnson"})

        assert result["_guide"]["page_dimensions"] == [[612.0, 792.0]]

    def test_bbox_format_in_guide(self):
        doc = self._make_doc()
        result = build_source_block(doc, fields={"name": "Sarah Johnson"})

        assert result["_guide"]["bbox_format"] == "[x0, y0, x1, y1]"

    def test_ungrounded_field(self):
        doc = self._make_doc()
        result = build_source_block(doc, fields={"missing": "NONEXISTENT"})

        assert result["fields"]["missing"]["grounded"] is False
        assert result["fields"]["missing"]["spans"] == []

    def test_field_evidence_adds_source_value_and_canonicalization(self):
        doc = self._make_doc(text="Mrs SAKUNTHALA 62Y/F")
        result = build_source_block(
            doc,
            fields={"sex": "Female"},
            field_evidence={
                "sex": {
                    "excerpt": "F",
                    "reasoning": "The source token F denotes female sex.",
                }
            },
        )

        field = result["fields"]["sex"]
        assert field["grounded"] is True
        assert field["source_value"] == "F"
        assert field["canonicalization"]["applied"] is True
        assert field["canonicalization"]["method"] == "llm_extraction"
        assert field["canonicalization"]["from"] == "F"
        assert field["canonicalization"]["to"] == "Female"
        assert "62Y/F" in field["excerpt"]


SAMPLE_PDF = Path(__file__).parent / "datasets" / "extract" / "sample_patient_vitals.pdf"


class TestBuildSourceBlockIntegration:
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_real_pdf_coordinates_match_pymupdf(self):
        """Verify _source spans match PyMuPDF search_for positions."""
        from mosaicx.documents.loader import load_document

        doc = load_document(SAMPLE_PDF)
        result = build_source_block(doc, fields={
            "patient_name": "Sarah Johnson",
            "patient_id": "PID-12345",
        })

        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        pdf = fitz.open(str(SAMPLE_PDF))
        page = pdf[0]
        page_h = doc.page_dimensions[0][1]

        for field_key, info in result["fields"].items():
            if not info["spans"]:
                continue
            span = info["spans"][0]
            x0, y0, x1, y1 = span["bbox"]
            our_rect = (x0, page_h - y1, x1, page_h - y0)

            search_rects = page.search_for(str(info["value"]))
            assert search_rects, f"PyMuPDF couldn't find {info['value']!r}"
            sr = search_rects[0]

            assert abs(our_rect[0] - sr.x0) < 15, (
                f"{field_key}: x0 off by {abs(our_rect[0] - sr.x0):.0f}"
            )
            assert abs(our_rect[1] - sr.y0) < 15, (
                f"{field_key}: y0 off by {abs(our_rect[1] - sr.y0):.0f}"
            )

        pdf.close()

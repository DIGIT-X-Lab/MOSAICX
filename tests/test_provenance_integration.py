"""Integration tests for source provenance end-to-end."""
from __future__ import annotations

import pytest
from pathlib import Path

SAMPLE_PDF = Path("tests/datasets/extract/sample_patient_vitals.pdf")
SAMPLE_TXT = Path("tests/datasets/extract/sample_patient_note.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_pdf():
    from mosaicx.documents.loader import load_document

    return load_document(SAMPLE_PDF)


def _load_txt():
    from mosaicx.documents.loader import load_document

    return load_document(SAMPLE_TXT)


# ---------------------------------------------------------------------------
# TestDeidentifyProvenance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
class TestDeidentifyProvenance:
    def test_deidentify_provenance_output_structure(self):
        """Load PDF, find a known name, enrich a simulated redaction map entry,
        verify the result has spans with page=1 and valid bbox coordinates."""
        from mosaicx.pipelines.provenance import enrich_redaction_map

        doc = _load_pdf()

        # Locate "Sarah Johnson" in the document text
        needle = "Sarah Johnson"
        start = doc.text.find(needle)
        assert start != -1, "Expected 'Sarah Johnson' in sample PDF"
        end = start + len(needle)

        # Simulate a redaction map entry as produced by the Deidentifier
        redaction_map = [
            {
                "original": needle,
                "replacement": "[NAME]",
                "start": start,
                "end": end,
                "phi_type": "NAME",
                "method": "llm",
            }
        ]

        enriched = enrich_redaction_map(doc, redaction_map)

        assert len(enriched) == 1
        entry = enriched[0]

        # Must have spans
        assert "spans" in entry
        assert isinstance(entry["spans"], list)
        assert len(entry["spans"]) >= 1

        # The span must be on page 1
        span = entry["spans"][0]
        assert span["page"] == 1

        # bbox values must be valid positive floats within page dimensions
        page_w, page_h = doc.page_dimensions[0]  # (612.0, 792.0)
        x0, y0, x1, y1 = span["bbox"]
        assert x0 >= 0.0
        assert y0 >= 0.0
        assert x1 > x0
        assert y1 > y0
        assert x1 <= page_w + 1  # small tolerance for floating point
        assert y1 <= page_h + 1

        # resolution must be "located"
        assert entry["resolution"] == "located"

        # excerpt must contain the original text
        assert "excerpt" in entry
        assert needle in entry["excerpt"]

    def test_text_block_invariant(self):
        """Verify doc.text[tb.start:tb.end] == tb.text for ALL TextBlocks."""
        doc = _load_pdf()

        assert len(doc.text_blocks) > 0, "Expected text blocks from native PDF load"

        violations = []
        for i, tb in enumerate(doc.text_blocks):
            actual = doc.text[tb.start : tb.end]
            if actual != tb.text:
                violations.append(
                    {
                        "index": i,
                        "tb_text": repr(tb.text),
                        "doc_slice": repr(actual),
                        "start": tb.start,
                        "end": tb.end,
                    }
                )

        assert violations == [], (
            f"TextBlock invariant violated for {len(violations)} block(s):\n"
            + "\n".join(str(v) for v in violations[:5])
        )


# ---------------------------------------------------------------------------
# TestLocateOnRealPDF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
class TestLocateOnRealPDF:
    def test_locate_known_text(self):
        """Load PDF, find 'Sarah Johnson', call locate_in_document, verify result."""
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _load_pdf()

        needle = "Sarah Johnson"
        start = doc.text.find(needle)
        assert start != -1, "Expected 'Sarah Johnson' in sample PDF"
        end = start + len(needle)

        result = locate_in_document(doc, start, end)

        assert result is not None
        assert len(result) >= 1

        span = result[0]
        assert span["page"] == 1

        x0, y0, x1, y1 = span["bbox"]
        assert x0 > 0.0
        assert y0 > 0.0
        assert x1 > x0
        assert y1 > y0

    def test_locate_returns_none_for_plain_text(self):
        """Loading a .txt file produces no TextBlocks; locate_in_document returns None."""
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _load_txt()

        # .txt files have no text_blocks
        assert doc.text_blocks == []

        result = locate_in_document(doc, 0, 10)
        assert result is None


# ---------------------------------------------------------------------------
# TestResolveOnRealPDF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
class TestResolveOnRealPDF:
    def test_resolve_provenance_on_pdf(self):
        """Load PDF, call resolve_provenance with known name, verify exact match."""
        from mosaicx.pipelines.provenance import resolve_provenance

        doc = _load_pdf()

        evidence = {"name": "Sarah Johnson"}
        result = resolve_provenance(doc, evidence)

        assert "name" in result
        entry = result["name"]

        # Must resolve exactly (text is verbatim in the document)
        assert entry["resolution"] == "exact"
        assert entry["excerpt"] == "Sarah Johnson"
        assert entry["start"] >= 0
        assert entry["end"] > entry["start"]

        # Must have at least one span with valid coordinates
        assert isinstance(entry["spans"], list)
        assert len(entry["spans"]) >= 1

        span = entry["spans"][0]
        assert span["page"] == 1
        x0, y0, x1, y1 = span["bbox"]
        assert x0 > 0.0
        assert y0 > 0.0
        assert x1 > x0
        assert y1 > y0

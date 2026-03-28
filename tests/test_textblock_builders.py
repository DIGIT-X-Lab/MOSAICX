# tests/test_textblock_builders.py
"""Tests for pypdfium2-based TextBlock builder in the document loader."""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_PDF = Path(__file__).parent / "datasets" / "extract" / "sample_patient_vitals.pdf"
SAMPLE_TXT = Path(__file__).parent / "datasets" / "extract" / "sample_patient_note.txt"


def _load_pdf():
    from mosaicx.documents.loader import load_document

    return load_document(SAMPLE_PDF)


class TestPypdfium2TextBlocks:
    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_loads_text_blocks_from_pdf(self):
        """doc.text_blocks should be non-empty for a native-text PDF."""
        doc = _load_pdf()
        assert doc.text_blocks, "text_blocks should be populated for a native PDF"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_blocks_cover_full_text(self):
        """Every non-whitespace char in doc.text should appear in at least one block."""
        doc = _load_pdf()
        # Build a set of all character offsets covered by blocks
        covered: set[int] = set()
        for tb in doc.text_blocks:
            for idx in range(tb.start, tb.end):
                covered.add(idx)

        for idx, ch in enumerate(doc.text):
            if not ch.isspace():
                assert idx in covered, (
                    f"Non-whitespace char {repr(ch)} at offset {idx} not covered by any TextBlock"
                )

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_blocks_sorted_by_start(self):
        """text_blocks must be in ascending order of start offset."""
        doc = _load_pdf()
        starts = [tb.start for tb in doc.text_blocks]
        assert starts == sorted(starts), "text_blocks are not sorted by start offset"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_blocks_have_valid_bboxes(self):
        """Every TextBlock must have x1 >= x0 and y1 >= y0."""
        doc = _load_pdf()
        for tb in doc.text_blocks:
            x0, y0, x1, y1 = tb.bbox
            assert x1 >= x0, f"Block {tb.text!r}: x1 ({x1}) < x0 ({x0})"
            assert y1 >= y0, f"Block {tb.text!r}: y1 ({y1}) < y0 ({y0})"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_page_dimensions_captured(self):
        """page_dimensions should be non-empty with positive width and height."""
        doc = _load_pdf()
        assert doc.page_dimensions, "page_dimensions should be populated for a native PDF"
        for w, h in doc.page_dimensions:
            assert w > 0, f"Page width {w} is not positive"
            assert h > 0, f"Page height {h} is not positive"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_block_text_matches_source(self):
        """CRITICAL: doc.text[tb.start:tb.end] must equal tb.text for every block."""
        doc = _load_pdf()
        for tb in doc.text_blocks:
            slice_text = doc.text[tb.start : tb.end]
            assert slice_text == tb.text, (
                f"Offset mismatch: doc.text[{tb.start}:{tb.end}]={slice_text!r} "
                f"but TextBlock.text={tb.text!r}"
            )

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_blocks_have_correct_pages(self):
        """Page numbers should be 1-indexed and within range."""
        doc = _load_pdf()
        page_count = doc.page_count or 1
        for tb in doc.text_blocks:
            assert tb.page >= 1, f"Block has page {tb.page} (expected >= 1)"
            assert tb.page <= page_count, (
                f"Block has page {tb.page} but document only has {page_count} pages"
            )

    def test_plain_text_has_no_text_blocks(self, tmp_path):
        """Plain .txt files should have empty text_blocks."""
        from mosaicx.documents.loader import load_document

        txt_file = tmp_path / "note.txt"
        txt_file.write_text("Patient presents with cough.")
        doc = load_document(txt_file)
        assert doc.text_blocks == [], "text files should have no TextBlocks"

    def test_plain_text_has_no_page_dimensions(self, tmp_path):
        """Plain .txt files should have empty page_dimensions."""
        from mosaicx.documents.loader import load_document

        txt_file = tmp_path / "note.txt"
        txt_file.write_text("Patient presents with cough.")
        doc = load_document(txt_file)
        assert doc.page_dimensions == [], "text files should have no page_dimensions"

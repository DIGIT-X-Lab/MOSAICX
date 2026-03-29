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


class TestPaddleOCRTextBlocks:
    def test_polygon_to_bbox_conversion(self):
        """Normal 4-point axis-aligned polygon -> correct (x0, y0, x1, y1)."""
        from mosaicx.documents.engines.paddleocr_engine import _polygon_to_bbox

        poly = [[10, 20], [90, 20], [90, 50], [10, 50]]
        assert _polygon_to_bbox(poly) == (10, 20, 90, 50)

    def test_polygon_to_bbox_rotated(self):
        """Rotated polygon -> axis-aligned bounding box."""
        from mosaicx.documents.engines.paddleocr_engine import _polygon_to_bbox

        # A diamond rotated 45 degrees: points not axis-aligned
        poly = [[50, 10], [90, 30], [50, 50], [10, 30]]
        assert _polygon_to_bbox(poly) == (10, 10, 90, 50)

    def test_build_textblocks_from_ocr_result(self):
        """2 texts + 2 polys -> 2 TextBlocks with correct offsets."""
        from mosaicx.documents.engines.paddleocr_engine import _build_ocr_textblocks

        rec_texts = ["Hello", "World"]
        dt_polys = [
            [[0, 0], [50, 0], [50, 10], [0, 10]],
            [[0, 15], [60, 15], [60, 25], [0, 25]],
        ]
        blocks = _build_ocr_textblocks(rec_texts, dt_polys, page_num=1, global_offset=0)

        assert len(blocks) == 2

        # First block: "Hello", offset 0..5
        assert blocks[0].text == "Hello"
        assert blocks[0].start == 0
        assert blocks[0].end == 5
        assert blocks[0].page == 1
        assert blocks[0].bbox == (0, 0, 50, 10)

        # Second block: "World", offset 6..11 (5 + 1 for "\n" separator)
        assert blocks[1].text == "World"
        assert blocks[1].start == 6
        assert blocks[1].end == 11
        assert blocks[1].page == 1
        assert blocks[1].bbox == (0, 15, 60, 25)

    def test_build_textblocks_empty_input(self):
        """Empty lists -> empty result."""
        from mosaicx.documents.engines.paddleocr_engine import _build_ocr_textblocks

        blocks = _build_ocr_textblocks([], [], page_num=1, global_offset=0)
        assert blocks == []

    def test_build_textblocks_mismatched_lengths(self):
        """3 texts + 2 polys -> 2 TextBlocks (skip missing polys)."""
        from mosaicx.documents.engines.paddleocr_engine import _build_ocr_textblocks

        rec_texts = ["Alpha", "Beta", "Gamma"]
        dt_polys = [
            [[0, 0], [40, 0], [40, 10], [0, 10]],
            [[0, 15], [35, 15], [35, 25], [0, 25]],
        ]
        blocks = _build_ocr_textblocks(rec_texts, dt_polys, page_num=2, global_offset=0)

        assert len(blocks) == 2
        assert blocks[0].text == "Alpha"
        assert blocks[1].text == "Beta"


class TestOffsetMapping:
    """Tests for _map_ocr_blocks_to_markdown in ppstructure_engine."""

    def test_simple_text_mapping(self):
        """OCR texts found sequentially in markdown get correct offsets."""
        from mosaicx.documents.engines.ppstructure_engine import (
            _map_ocr_blocks_to_markdown,
        )

        markdown = "Hello World"
        rec_texts = ["Hello", "World"]
        dt_polys = [
            [[0, 0], [50, 0], [50, 10], [0, 10]],
            [[60, 0], [120, 0], [120, 10], [60, 10]],
        ]
        blocks = _map_ocr_blocks_to_markdown(
            markdown, rec_texts, dt_polys, page_num=1, global_offset=0
        )
        assert len(blocks) == 2
        assert blocks[0].text == "Hello"
        assert blocks[0].start == 0
        assert blocks[0].end == 5
        assert blocks[0].bbox == (0, 0, 50, 10)
        assert blocks[1].text == "World"
        assert blocks[1].start == 6
        assert blocks[1].end == 11

    def test_table_markdown_mapping(self):
        """OCR texts found inside markdown table rows."""
        from mosaicx.documents.engines.ppstructure_engine import (
            _map_ocr_blocks_to_markdown,
        )

        markdown = "| Name | Value |\n|---|---|\n| BP | 120/80 |"
        rec_texts = ["Name", "Value", "BP", "120/80"]
        dt_polys = [
            [[0, 0], [40, 0], [40, 10], [0, 10]],
            [[50, 0], [90, 0], [90, 10], [50, 10]],
            [[0, 20], [30, 20], [30, 30], [0, 30]],
            [[50, 20], [100, 20], [100, 30], [50, 30]],
        ]
        blocks = _map_ocr_blocks_to_markdown(
            markdown, rec_texts, dt_polys, page_num=1, global_offset=0
        )
        assert len(blocks) == 4
        for b in blocks:
            assert markdown[b.start : b.end] == b.text, (
                f"Mismatch: markdown[{b.start}:{b.end}]={markdown[b.start:b.end]!r} "
                f"!= {b.text!r}"
            )

    def test_global_offset_applied(self):
        """global_offset shifts all start/end positions."""
        from mosaicx.documents.engines.ppstructure_engine import (
            _map_ocr_blocks_to_markdown,
        )

        markdown = "Hello"
        rec_texts = ["Hello"]
        dt_polys = [[[0, 0], [50, 0], [50, 10], [0, 10]]]
        blocks = _map_ocr_blocks_to_markdown(
            markdown, rec_texts, dt_polys, page_num=2, global_offset=100
        )
        assert blocks[0].start == 100
        assert blocks[0].end == 105
        assert blocks[0].page == 2

    def test_missing_text_uses_fallback(self):
        """If an OCR text is not found in markdown, still produces a block."""
        from mosaicx.documents.engines.ppstructure_engine import (
            _map_ocr_blocks_to_markdown,
        )

        markdown = "Hello World"
        rec_texts = ["Hello", "MISSING_TEXT"]
        dt_polys = [
            [[0, 0], [50, 0], [50, 10], [0, 10]],
            [[60, 0], [120, 0], [120, 10], [60, 10]],
        ]
        blocks = _map_ocr_blocks_to_markdown(
            markdown, rec_texts, dt_polys, page_num=1, global_offset=0
        )
        assert len(blocks) == 2
        assert blocks[0].text == "Hello"

    def test_empty_inputs(self):
        """Empty rec_texts -> empty blocks."""
        from mosaicx.documents.engines.ppstructure_engine import (
            _map_ocr_blocks_to_markdown,
        )

        blocks = _map_ocr_blocks_to_markdown("", [], [], page_num=1, global_offset=0)
        assert blocks == []

    def test_mismatched_poly_count(self):
        """More texts than polys -> only produce blocks for paired entries."""
        from mosaicx.documents.engines.ppstructure_engine import (
            _map_ocr_blocks_to_markdown,
        )

        markdown = "Alpha Beta Gamma"
        rec_texts = ["Alpha", "Beta", "Gamma"]
        dt_polys = [
            [[0, 0], [40, 0], [40, 10], [0, 10]],
            [[50, 0], [80, 0], [80, 10], [50, 10]],
        ]
        blocks = _map_ocr_blocks_to_markdown(
            markdown, rec_texts, dt_polys, page_num=1, global_offset=0
        )
        assert len(blocks) == 2
        assert blocks[0].text == "Alpha"
        assert blocks[1].text == "Beta"

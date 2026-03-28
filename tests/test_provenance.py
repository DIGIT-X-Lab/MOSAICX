"""Tests for TextBlock model, LoadedDocument extensions, and locate_in_document."""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Task 1: TextBlock and LoadedDocument extensions
# ---------------------------------------------------------------------------


class TestTextBlock:
    def test_create_text_block(self):
        from mosaicx.documents.models import TextBlock

        block = TextBlock(
            text="Patient has bilateral opacity.",
            start=0,
            end=29,
            page=1,
            bbox=(10.0, 20.0, 200.0, 35.0),
        )
        assert block.text == "Patient has bilateral opacity."
        assert block.start == 0
        assert block.end == 29
        assert block.page == 1
        assert block.bbox == (10.0, 20.0, 200.0, 35.0)

    def test_text_block_fields_are_required(self):
        from mosaicx.documents.models import TextBlock

        # All five fields are required — omitting any should raise TypeError
        with pytest.raises(TypeError):
            TextBlock(text="hello", start=0, end=5, page=1)  # missing bbox

        with pytest.raises(TypeError):
            TextBlock(text="hello", start=0, end=5, bbox=(0, 0, 1, 1))  # missing page

        with pytest.raises(TypeError):
            TextBlock(text="hello", start=0, page=1, bbox=(0, 0, 1, 1))  # missing end

        with pytest.raises(TypeError):
            TextBlock(text="hello", end=5, page=1, bbox=(0, 0, 1, 1))  # missing start

        with pytest.raises(TypeError):
            TextBlock(start=0, end=5, page=1, bbox=(0, 0, 1, 1))  # missing text


class TestLoadedDocumentExtensions:
    def test_loaded_document_has_text_blocks(self):
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(
            text="Hello world",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
        )
        assert hasattr(doc, "text_blocks")
        assert doc.text_blocks == []

    def test_loaded_document_has_page_dimensions(self):
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(
            text="Hello world",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
        )
        assert hasattr(doc, "page_dimensions")
        assert doc.page_dimensions == []

    def test_loaded_document_with_text_blocks(self):
        from mosaicx.documents.models import LoadedDocument, TextBlock

        blocks = [
            TextBlock(text="First line.", start=0, end=11, page=1, bbox=(10.0, 10.0, 100.0, 25.0)),
            TextBlock(text="Second line.", start=12, end=24, page=1, bbox=(10.0, 30.0, 100.0, 45.0)),
        ]
        doc = LoadedDocument(
            text="First line.\nSecond line.",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            text_blocks=blocks,
            page_dimensions=[(595.0, 842.0)],
        )
        assert len(doc.text_blocks) == 2
        assert doc.text_blocks[0].text == "First line."
        assert doc.text_blocks[1].start == 12
        assert doc.page_dimensions == [(595.0, 842.0)]

    def test_page_result_has_text_blocks(self):
        from mosaicx.documents.models import PageResult, TextBlock

        block = TextBlock(
            text="Liver unremarkable.",
            start=0,
            end=19,
            page=1,
            bbox=(10.0, 50.0, 200.0, 65.0),
        )
        page = PageResult(
            page_number=1,
            text="Liver unremarkable.",
            engine="surya",
            confidence=0.92,
            text_blocks=[block],
        )
        assert len(page.text_blocks) == 1
        assert page.text_blocks[0].text == "Liver unremarkable."

    def test_page_result_text_blocks_default_empty(self):
        from mosaicx.documents.models import PageResult

        page = PageResult(
            page_number=1,
            text="Text",
            engine="surya",
            confidence=0.9,
        )
        assert page.text_blocks == []


# ---------------------------------------------------------------------------
# Task 2: locate_in_document()
# ---------------------------------------------------------------------------


def _make_doc(blocks: list[dict]) -> "LoadedDocument":
    """Helper: build a LoadedDocument with TextBlock list from dicts."""
    from mosaicx.documents.models import LoadedDocument, TextBlock

    text_blocks = [
        TextBlock(
            text=b["text"],
            start=b["start"],
            end=b["end"],
            page=b["page"],
            bbox=tuple(b["bbox"]),
        )
        for b in blocks
    ]
    full_text = " ".join(b["text"] for b in blocks)
    return LoadedDocument(
        text=full_text,
        source_path=Path("/tmp/doc.pdf"),
        format="pdf",
        text_blocks=text_blocks,
    )


class TestLocateInDocument:
    def test_single_block_exact_match(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "Pleural effusion noted.", "start": 0, "end": 23, "page": 1, "bbox": [10, 20, 200, 35]},
        ])
        result = locate_in_document(doc, 0, 23)
        assert result is not None
        assert len(result) == 1
        assert result[0]["page"] == 1
        assert result[0]["bbox"] == (10, 20, 200, 35)
        assert result[0]["start"] == 0
        assert result[0]["end"] == 23

    def test_range_spanning_multiple_blocks_same_page(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "First block.", "start": 0, "end": 12, "page": 1, "bbox": [10, 10, 100, 25]},
            {"text": "Second block.", "start": 13, "end": 26, "page": 1, "bbox": [10, 30, 110, 45]},
            {"text": "Third block.", "start": 27, "end": 39, "page": 1, "bbox": [10, 50, 100, 65]},
        ])
        # Range covers first two blocks
        result = locate_in_document(doc, 0, 26)
        assert result is not None
        assert len(result) == 1  # same page => one entry
        entry = result[0]
        assert entry["page"] == 1
        # Union bbox: x0=min(10,10)=10, y0=min(10,30)=10, x1=max(100,110)=110, y1=max(25,45)=45
        assert entry["bbox"] == (10, 10, 110, 45)
        assert entry["start"] == 0
        assert entry["end"] == 26

    def test_range_spanning_multiple_pages(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "Page one text.", "start": 0, "end": 14, "page": 1, "bbox": [10, 700, 200, 715]},
            {"text": "Page two text.", "start": 15, "end": 29, "page": 2, "bbox": [10, 10, 200, 25]},
        ])
        result = locate_in_document(doc, 0, 29)
        assert result is not None
        assert len(result) == 2
        pages = {r["page"]: r for r in result}
        assert 1 in pages
        assert 2 in pages
        assert pages[1]["bbox"] == (10, 700, 200, 715)
        assert pages[2]["bbox"] == (10, 10, 200, 25)

    def test_no_text_blocks_returns_none(self):
        from mosaicx.documents.models import LoadedDocument
        from mosaicx.pipelines.provenance import locate_in_document

        doc = LoadedDocument(
            text="Some text without blocks.",
            source_path=Path("/tmp/empty.pdf"),
            format="pdf",
        )
        assert doc.text_blocks == []
        result = locate_in_document(doc, 0, 10)
        assert result is None

    def test_range_outside_blocks_returns_none(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "Block text.", "start": 50, "end": 61, "page": 1, "bbox": [10, 10, 100, 25]},
        ])
        # Query range is entirely before the block
        result = locate_in_document(doc, 0, 20)
        assert result is None

    def test_partial_overlap_with_blocks(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "Alpha block.", "start": 0, "end": 12, "page": 1, "bbox": [10, 10, 100, 25]},
            {"text": "Beta block.", "start": 20, "end": 31, "page": 1, "bbox": [10, 30, 100, 45]},
            {"text": "Gamma block.", "start": 40, "end": 52, "page": 2, "bbox": [10, 10, 100, 25]},
        ])
        # Range [5, 25] overlaps Alpha (0-12) and Beta (20-31), not Gamma (40-52)
        result = locate_in_document(doc, 5, 25)
        assert result is not None
        assert len(result) == 1  # both overlapping blocks are on page 1
        assert result[0]["page"] == 1
        # Union of Alpha and Beta bboxes
        assert result[0]["bbox"] == (10, 10, 100, 45)
        assert result[0]["start"] == 5
        assert result[0]["end"] == 25

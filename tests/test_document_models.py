# tests/test_document_models.py
"""Tests for document data models."""

import pytest
from pathlib import Path


class TestPageResult:
    def test_construction(self):
        from mosaicx.documents.models import PageResult

        page = PageResult(
            page_number=1,
            text="Patient presents with cough.",
            engine="surya",
            confidence=0.85,
        )
        assert page.page_number == 1
        assert page.engine == "surya"
        assert page.confidence == 0.85
        assert page.layout_html is None

    def test_with_layout_html(self):
        from mosaicx.documents.models import PageResult

        page = PageResult(
            page_number=1,
            text="Test",
            engine="chandra",
            confidence=0.9,
            layout_html="<div>Test</div>",
        )
        assert page.layout_html == "<div>Test</div>"


class TestLoadedDocumentNewFields:
    def test_ocr_metadata_defaults(self):
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(
            text="Hello",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
        )
        assert doc.ocr_engine_used is None
        assert doc.ocr_confidence is None
        assert doc.quality_warning is False
        assert doc.pages == []

    def test_ocr_metadata_populated(self):
        from mosaicx.documents.models import LoadedDocument, PageResult

        pages = [
            PageResult(page_number=1, text="Page 1", engine="surya", confidence=0.9),
            PageResult(page_number=2, text="Page 2", engine="chandra", confidence=0.7),
        ]
        doc = LoadedDocument(
            text="Page 1\nPage 2",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            page_count=2,
            ocr_engine_used="mixed",
            ocr_confidence=0.8,
            quality_warning=False,
            pages=pages,
        )
        assert doc.ocr_engine_used == "mixed"
        assert len(doc.pages) == 2
        assert doc.pages[0].engine == "surya"
        assert doc.pages[1].engine == "chandra"

    def test_backward_compat_char_count_is_empty(self):
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(text="abc", source_path=Path("/tmp/x.txt"), format="txt")
        assert doc.char_count == 3
        assert doc.is_empty is False

        empty = LoadedDocument(text="", source_path=Path("/tmp/x.txt"), format="txt")
        assert empty.is_empty is True


class TestDocumentLoadError:
    def test_is_exception(self):
        from mosaicx.documents.models import DocumentLoadError

        err = DocumentLoadError("Corrupted PDF")
        assert isinstance(err, Exception)
        assert str(err) == "Corrupted PDF"

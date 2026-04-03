# tests/test_loader_orchestrator.py
"""Tests for the document loader orchestrator (dual-engine)."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from mosaicx.documents.models import PageResult, LoadedDocument, DocumentLoadError


def _make_page(page_num, text, engine, confidence):
    return PageResult(page_number=page_num, text=text, engine=engine, confidence=confidence)


class TestTextFileBypass:
    """Text files should skip OCR entirely."""

    def test_txt_no_ocr(self, tmp_path):
        from mosaicx.documents.loader import load_document

        f = tmp_path / "report.txt"
        f.write_text("Patient presents with cough.")
        doc = load_document(f)
        assert doc.ocr_engine_used is None
        assert doc.pages == []
        assert "Patient presents" in doc.text

    def test_md_no_ocr(self, tmp_path):
        from mosaicx.documents.loader import load_document

        f = tmp_path / "report.md"
        f.write_text("# Findings\nNormal.")
        doc = load_document(f)
        assert doc.ocr_engine_used is None


class TestErrorHandling:
    def test_missing_file_raises(self):
        from mosaicx.documents.loader import load_document
        from mosaicx.documents.models import DocumentLoadError

        with pytest.raises((FileNotFoundError, DocumentLoadError)):
            load_document(Path("/nonexistent/file.pdf"))

    def test_unsupported_format_raises(self, tmp_path):
        from mosaicx.documents.loader import load_document

        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(f)


class TestImageFormats:
    def test_image_formats_accepted(self):
        from mosaicx.documents.engines.base import SUPPORTED_FORMATS

        for ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
            assert ext in SUPPORTED_FORMATS

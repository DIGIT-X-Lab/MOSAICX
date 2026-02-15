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


class TestDualEngineOrchestration:
    """Test the parallel engine dispatch and quality scoring."""

    def test_picks_higher_quality_engine(self, tmp_path):
        from mosaicx.documents.loader import _pick_best_pages

        surya_pages = [_make_page(1, "Good medical text with nodule", "surya", 0.8)]
        chandra_pages = [_make_page(1, "G00d m3d1c@l t3xt", "chandra", 0.3)]

        winners = _pick_best_pages(surya_pages, chandra_pages)
        assert len(winners) == 1
        assert winners[0].engine == "surya"

    def test_mixed_page_winners(self):
        from mosaicx.documents.loader import _pick_best_pages

        surya_pages = [
            _make_page(1, "Good typed text with findings and impression", "surya", 0.9),
            _make_page(2, "@#$ garbled", "surya", 0.1),
        ]
        chandra_pages = [
            _make_page(1, "OK text", "chandra", 0.5),
            _make_page(2, "Handwritten note about patient diagnosis", "chandra", 0.8),
        ]

        winners = _pick_best_pages(surya_pages, chandra_pages)
        assert winners[0].engine == "surya"   # page 1: surya better
        assert winners[1].engine == "chandra" # page 2: chandra better

    def test_quality_warning_when_both_low(self):
        from mosaicx.documents.loader import _pick_best_pages, _assemble_document

        surya_pages = [_make_page(1, "@@@", "surya", 0.1)]
        chandra_pages = [_make_page(1, "###", "chandra", 0.1)]

        winners = _pick_best_pages(surya_pages, chandra_pages, threshold=0.6)
        doc = _assemble_document(
            winners=winners,
            source_path=Path("/tmp/test.pdf"),
            fmt="pdf",
            threshold=0.6,
        )
        assert doc.quality_warning is True


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

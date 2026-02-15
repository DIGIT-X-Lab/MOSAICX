# tests/test_engine_base.py
"""Tests for OCR engine protocol and image utilities."""

import pytest
from pathlib import Path


class TestOCREngineProtocol:
    def test_protocol_is_importable(self):
        from mosaicx.documents.engines.base import OCREngine
        assert OCREngine is not None

    def test_protocol_defines_ocr_pages(self):
        from mosaicx.documents.engines.base import OCREngine
        import inspect
        members = dict(inspect.getmembers(OCREngine))
        assert "ocr_pages" in members or hasattr(OCREngine, "ocr_pages")


class TestPdfToImages:
    def test_function_exists(self):
        from mosaicx.documents.engines.base import pdf_to_images
        assert callable(pdf_to_images)

    def test_image_to_pages(self, tmp_path):
        from mosaicx.documents.engines.base import image_to_pages
        from PIL import Image

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        pages = image_to_pages(img_path)
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)


class TestSupportedFormats:
    def test_supported_formats_exported(self):
        from mosaicx.documents.engines.base import SUPPORTED_FORMATS
        assert ".pdf" in SUPPORTED_FORMATS
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".png" in SUPPORTED_FORMATS
        assert ".tiff" in SUPPORTED_FORMATS
        assert ".txt" in SUPPORTED_FORMATS

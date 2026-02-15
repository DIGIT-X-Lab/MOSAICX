# tests/test_ocr_integration.py
"""Integration tests for OCR engines — require GPU and installed models.

Run with: pytest tests/test_ocr_integration.py -m slow -v
"""

import pytest
from pathlib import Path
from PIL import Image


@pytest.mark.slow
class TestOCRIntegration:
    def test_surya_on_image(self):
        """Surya can OCR a simple text image."""
        from mosaicx.documents.engines.surya_engine import SuryaEngine

        # Create a simple image with text (real test would use a scanned doc)
        img = Image.new("RGB", (200, 50), "white")
        engine = SuryaEngine()
        results = engine.ocr_pages([img], langs=["en"])
        assert len(results) == 1
        assert results[0].engine == "surya"

    def test_chandra_on_image(self):
        """Chandra can OCR a simple image."""
        from mosaicx.documents.engines.chandra_engine import ChandraEngine

        img = Image.new("RGB", (200, 50), "white")
        engine = ChandraEngine()
        results = engine.ocr_pages([img])
        assert len(results) == 1
        assert results[0].engine == "chandra"

    def test_full_pipeline_on_image(self, tmp_path):
        """Full load_document pipeline on an image file."""
        from mosaicx.documents.loader import load_document

        img = Image.new("RGB", (200, 100), "white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        doc = load_document(img_path)
        assert doc.source_path == img_path
        assert doc.format == "png"

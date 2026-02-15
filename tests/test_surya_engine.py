# tests/test_surya_engine.py
"""Tests for the Surya OCR engine wrapper."""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from mosaicx.documents.models import PageResult


class TestSuryaEngine:
    def test_implements_protocol(self):
        from mosaicx.documents.engines.base import OCREngine
        from mosaicx.documents.engines.surya_engine import SuryaEngine

        engine = SuryaEngine.__new__(SuryaEngine)
        assert isinstance(engine, OCREngine)

    def test_ocr_pages_returns_page_results(self):
        from mosaicx.documents.engines.surya_engine import SuryaEngine

        # Mock surya internals
        with patch("mosaicx.documents.engines.surya_engine._get_surya_pipeline") as mock_pipeline:
            mock_result = MagicMock()
            mock_result.text_lines = [
                MagicMock(text="Patient presents with cough."),
                MagicMock(text="CT shows 5mm nodule."),
            ]
            mock_result.confidence = 0.92
            mock_pipeline.return_value.return_value = [mock_result]

            engine = SuryaEngine()
            images = [Image.new("RGB", (100, 100), "white")]
            results = engine.ocr_pages(images, langs=["en"])

            assert len(results) == 1
            assert isinstance(results[0], PageResult)
            assert results[0].engine == "surya"
            assert "Patient presents" in results[0].text

    def test_empty_image_list(self):
        from mosaicx.documents.engines.surya_engine import SuryaEngine

        with patch("mosaicx.documents.engines.surya_engine._get_surya_pipeline"):
            engine = SuryaEngine()
            results = engine.ocr_pages([], langs=["en"])
            assert results == []

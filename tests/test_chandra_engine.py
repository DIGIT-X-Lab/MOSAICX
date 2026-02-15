# tests/test_chandra_engine.py
"""Tests for the Chandra OCR engine wrapper."""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from mosaicx.documents.models import PageResult


class TestChandraEngine:
    def test_implements_protocol(self):
        from mosaicx.documents.engines.base import OCREngine
        from mosaicx.documents.engines.chandra_engine import ChandraEngine

        engine = ChandraEngine.__new__(ChandraEngine)
        assert isinstance(engine, OCREngine)

    def test_ocr_pages_returns_page_results(self):
        from mosaicx.documents.engines.chandra_engine import ChandraEngine

        with patch("mosaicx.documents.engines.chandra_engine._get_chandra_manager") as mock_mgr:
            mock_result = MagicMock()
            mock_result.markdown = "Patient presents with cough.\n\n5mm nodule in RUL."
            mock_result.html = "<div>Patient presents with cough.</div>"
            mock_mgr.return_value.generate.return_value = [mock_result]

            engine = ChandraEngine()
            images = [Image.new("RGB", (100, 100), "white")]
            results = engine.ocr_pages(images)

            assert len(results) == 1
            assert isinstance(results[0], PageResult)
            assert results[0].engine == "chandra"
            assert "Patient presents" in results[0].text
            assert results[0].layout_html is not None

    def test_auto_backend_detection(self):
        from mosaicx.documents.engines.chandra_engine import _detect_backend

        backend = _detect_backend()
        assert backend in ("vllm", "hf")

    def test_empty_image_list(self):
        from mosaicx.documents.engines.chandra_engine import ChandraEngine

        with patch("mosaicx.documents.engines.chandra_engine._get_chandra_manager"):
            engine = ChandraEngine()
            results = engine.ocr_pages([])
            assert results == []

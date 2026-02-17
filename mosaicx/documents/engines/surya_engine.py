# mosaicx/documents/engines/surya_engine.py
"""Surya OCR engine wrapper.

Surya provides fast layout-aware OCR using traditional pipeline
(detection -> recognition). Works on CPU and GPU (CUDA/MPS).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from PIL import Image

from ..models import PageResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_surya_pipeline():
    """Lazily load and cache the Surya OCR pipeline."""
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    foundation = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation)

    def run_ocr(images: list[Image.Image], langs: list[str] | None = None):
        return rec_predictor(images, det_predictor=det_predictor)

    return run_ocr


class SuryaEngine:
    """OCR engine backed by Surya."""

    def ocr_pages(
        self,
        images: list[Image.Image],
        langs: list[str] | None = None,
    ) -> list[PageResult]:
        if not images:
            return []

        try:
            pipeline = _get_surya_pipeline()
            results = pipeline(images, langs=langs)
        except Exception:
            logger.exception("Surya OCR failed")
            return [
                PageResult(page_number=i + 1, text="", engine="surya", confidence=0.0)
                for i in range(len(images))
            ]

        page_results = []
        for i, result in enumerate(results):
            lines = [line.text for line in result.text_lines]
            text = "\n".join(lines)
            conf = getattr(result, "confidence", 0.8)
            page_results.append(
                PageResult(
                    page_number=i + 1,
                    text=text,
                    engine="surya",
                    confidence=float(conf),
                )
            )
        return page_results

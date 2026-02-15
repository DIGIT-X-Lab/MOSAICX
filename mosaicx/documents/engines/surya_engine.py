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
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    def run_ocr(images: list[Image.Image], langs: list[str] | None = None):
        from surya.pipeline import OCRPipeline
        pipeline = OCRPipeline(det_predictor=det_predictor, rec_predictor=rec_predictor)
        return pipeline(images, langs=langs or ["en"])

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

        pipeline = _get_surya_pipeline()
        try:
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

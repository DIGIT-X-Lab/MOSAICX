# mosaicx/documents/engines/paddleocr_engine.py
"""PaddleOCR engine wrapper.

PaddleOCR provides fast, accurate OCR for printed text in Latin-script
languages. Uses lang="german" which covers English + German + other
Latin scripts.
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache

from PIL import Image

from ..models import PageResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_paddleocr():
    """Lazily load and cache the PaddleOCR instance."""
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    from paddleocr import PaddleOCR

    return PaddleOCR(lang="german")


class PaddleOCREngine:
    """OCR engine backed by PaddleOCR."""

    def ocr_pages(
        self,
        images: list[Image.Image],
        langs: list[str] | None = None,
    ) -> list[PageResult]:
        if not images:
            return []

        ocr = _get_paddleocr()
        page_results: list[PageResult] = []

        for i, img in enumerate(images):
            try:
                # PaddleOCR works best with file paths; save PIL image to
                # a temporary file and pass the path.
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp:
                    img.save(tmp, format="PNG")
                    tmp_path = tmp.name

                try:
                    results = list(ocr.predict(tmp_path))
                finally:
                    os.unlink(tmp_path)

                if results:
                    page_data = results[0]
                    res = page_data.json.get("res", {})
                    rec_texts = res.get("rec_texts", [])
                    rec_scores = res.get("rec_scores", [])

                    text = "\n".join(rec_texts)
                    confidence = (
                        sum(rec_scores) / len(rec_scores)
                        if rec_scores
                        else 0.0
                    )
                else:
                    text = ""
                    confidence = 0.0

                page_results.append(
                    PageResult(
                        page_number=i + 1,
                        text=text,
                        engine="paddleocr",
                        confidence=round(confidence, 4),
                    )
                )
            except Exception:
                logger.exception("PaddleOCR failed on page %d", i + 1)
                page_results.append(
                    PageResult(
                        page_number=i + 1,
                        text="",
                        engine="paddleocr",
                        confidence=0.0,
                    )
                )

        return page_results

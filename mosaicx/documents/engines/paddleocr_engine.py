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

from ..models import PageResult, TextBlock

logger = logging.getLogger(__name__)


def _polygon_to_bbox(
    poly: list[list[float]],
) -> tuple[float, float, float, float]:
    """Convert a 4-point polygon to an axis-aligned bounding box.

    Args:
        poly: List of [x, y] corner points (typically 4 corners).

    Returns:
        (x0, y0, x1, y1) axis-aligned bounding box.
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _build_ocr_textblocks(
    rec_texts: list[str],
    dt_polys: list[list[list[float]]],
    page_num: int,
    global_offset: int,
) -> list[TextBlock]:
    """Build TextBlock objects from PaddleOCR recognition results.

    Character offsets are computed from ``"\\n".join(rec_texts)``, so each
    text block is separated by a single newline character.  If ``dt_polys`` is
    shorter than ``rec_texts``, only the entries that have a corresponding
    polygon are included.

    Args:
        rec_texts: Recognised text strings, one per detection box.
        dt_polys:  Polygon coordinates, one per detection box.
        page_num:  1-indexed page number for the resulting TextBlocks.
        global_offset: Character offset of the first character of this page
            within the full document text (used when assembling multi-page
            documents).

    Returns:
        List of TextBlock objects, one per paired (text, polygon) entry.
    """
    blocks: list[TextBlock] = []
    cumulative = global_offset

    for idx, text in enumerate(rec_texts):
        if idx >= len(dt_polys):
            break
        poly = dt_polys[idx]
        bbox = _polygon_to_bbox(poly)
        start = cumulative
        end = start + len(text)
        blocks.append(
            TextBlock(
                text=text,
                start=start,
                end=end,
                page=page_num,
                bbox=bbox,
            )
        )
        # Advance by text length + 1 for the "\n" separator
        cumulative = end + 1

    return blocks


@lru_cache(maxsize=1)
def _get_paddleocr():
    """Lazily load and cache the PaddleOCR instance."""
    import io
    import sys
    import warnings

    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    # Suppress noisy PaddlePaddle and requests warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="paddleocr")
    warnings.filterwarnings("ignore", message=".*urllib3.*chardet.*charset_normalizer.*")
    warnings.filterwarnings("ignore", message=".*No ccache found.*")

    # Redirect paddle's verbose logging to WARNING+
    for name in ("paddle", "paddleocr", "paddlex", "ppocr"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # PaddleOCR prints "Creating model" and cache messages directly via
    # print() — silence them by redirecting stdout/stderr during init.
    _out, _err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(lang="german")
    finally:
        sys.stdout, sys.stderr = _out, _err

    return ocr


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
                    import io
                    import sys

                    _out, _err = sys.stdout, sys.stderr
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    try:
                        results = list(ocr.predict(tmp_path))
                    finally:
                        sys.stdout, sys.stderr = _out, _err
                finally:
                    os.unlink(tmp_path)

                if results:
                    page_data = results[0]
                    res = page_data.json.get("res", {})
                    rec_texts = res.get("rec_texts", [])
                    rec_scores = res.get("rec_scores", [])
                    dt_polys = res.get("dt_polys", [])

                    text = "\n".join(rec_texts)
                    confidence = (
                        sum(rec_scores) / len(rec_scores)
                        if rec_scores
                        else 0.0
                    )
                    ocr_blocks = _build_ocr_textblocks(
                        rec_texts, dt_polys, page_num=i + 1, global_offset=0
                    )
                else:
                    text = ""
                    confidence = 0.0
                    ocr_blocks = []

                page_results.append(
                    PageResult(
                        page_number=i + 1,
                        text=text,
                        engine="paddleocr",
                        confidence=round(confidence, 4),
                        text_blocks=ocr_blocks,
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

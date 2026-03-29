# mosaicx/documents/engines/ppstructure_engine.py
"""PPStructureV3 layout-aware OCR engine.

Replaces basic PaddleOCR with PPStructureV3, which adds layout detection
and table structure recognition. Accepts file paths directly (PDF, images)
and returns Markdown with tables preserved.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

from ..models import PageResult, TextBlock

logger = logging.getLogger(__name__)


def _polygon_to_bbox(
    poly: list[list[float]],
) -> tuple[float, float, float, float]:
    """Convert a 4-point polygon to an axis-aligned bounding box."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _map_ocr_blocks_to_markdown(
    markdown_text: str,
    rec_texts: list[str],
    dt_polys: list[list[list[float]]],
    page_num: int,
    global_offset: int,
) -> list[TextBlock]:
    """Build TextBlocks with start/end offsets into the Markdown text.

    Each ``rec_text`` is searched for sequentially in ``markdown_text``.
    If found, the TextBlock's ``start``/``end`` point to the match position
    in the Markdown (plus ``global_offset``).  If not found, the block still
    gets created with a best-effort sequential offset so redaction/provenance
    degrade gracefully rather than crash.

    Parameters
    ----------
    markdown_text : The Markdown string for this page (from PPStructureV3).
    rec_texts     : Per-line recognized text strings from overall_ocr_res.
    dt_polys      : Per-line 4-point polygons from overall_ocr_res.
    page_num      : 1-indexed page number.
    global_offset : Character offset of this page in the full document text.
    """
    blocks: list[TextBlock] = []
    search_start = 0

    for idx, text in enumerate(rec_texts):
        if idx >= len(dt_polys):
            break
        pos = markdown_text.find(text, search_start)
        if pos >= 0:
            start = pos
            search_start = pos + len(text)
        else:
            # Fallback: place after the last block's end.  Redaction bbox
            # is still correct (from dt_polys); only the text offset is
            # approximate.
            start = search_start

        bbox = _polygon_to_bbox(dt_polys[idx])
        blocks.append(
            TextBlock(
                text=text,
                start=global_offset + start,
                end=global_offset + start + len(text),
                page=page_num,
                bbox=bbox,
            )
        )

    return blocks


@lru_cache(maxsize=1)
def _get_ppstructure(lang: str | None = None):
    """Lazily load and cache the PPStructureV3 instance."""
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="paddleocr"
    )
    warnings.filterwarnings(
        "ignore", message=".*urllib3.*chardet.*charset_normalizer.*"
    )
    warnings.filterwarnings("ignore", message=".*No ccache found.*")

    for name in ("paddle", "paddleocr", "paddlex", "ppocr"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _out, _err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        from paddleocr import PPStructureV3

        engine = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_table_recognition=True,
            use_region_detection=True,
            lang=lang,
        )
    finally:
        sys.stdout, sys.stderr = _out, _err

    return engine


class PPStructureEngine:
    """Layout-aware OCR engine backed by PaddleOCR PPStructureV3."""

    def __init__(self, lang: str | None = None):
        self._lang = lang

    def process_file(self, path: Path) -> list[PageResult]:
        """Process a document file and return one PageResult per page."""
        engine = _get_ppstructure(self._lang)

        _out, _err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            results = list(engine.predict(str(path)))
        finally:
            sys.stdout, sys.stderr = _out, _err

        if not results:
            return [
                PageResult(
                    page_number=1,
                    text="",
                    engine="ppstructure",
                    confidence=0.0,
                )
            ]

        page_results: list[PageResult] = []
        for page_result in results:
            page_index = page_result["page_index"]
            page_num = page_index + 1

            # Extract Markdown text
            markdown_data = page_result.markdown
            if isinstance(markdown_data, dict):
                markdown_text = markdown_data.get("markdown_texts", "")
            else:
                markdown_text = getattr(
                    markdown_data, "markdown_texts", ""
                )
            if not isinstance(markdown_text, str):
                markdown_text = str(markdown_text) if markdown_text else ""

            # Extract raw OCR data for TextBlock construction
            ocr_res = page_result["overall_ocr_res"]
            if isinstance(ocr_res, dict):
                rec_texts = ocr_res.get("rec_texts", [])
                rec_scores = ocr_res.get("rec_scores", [])
                dt_polys = ocr_res.get("dt_polys", [])
            else:
                rec_texts = getattr(ocr_res, "rec_texts", [])
                rec_scores = getattr(ocr_res, "rec_scores", [])
                dt_polys = getattr(ocr_res, "dt_polys", [])

            # Build TextBlocks with offsets into the Markdown text
            text_blocks = _map_ocr_blocks_to_markdown(
                markdown_text,
                rec_texts,
                dt_polys,
                page_num=page_num,
                global_offset=0,
            )

            # Compute average confidence
            confidence = (
                sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
            )

            # Collect HTML from table blocks
            parsing_res = page_result["parsing_res_list"]
            table_htmls: list[str] = []
            for block in parsing_res:
                label = getattr(block, "label", None)
                if label is None and isinstance(block, dict):
                    label = block.get("label")
                if label == "table":
                    content = getattr(block, "content", None)
                    if content is None and isinstance(block, dict):
                        content = block.get("content")
                    if content:
                        table_htmls.append(content)

            layout_html = "\n".join(table_htmls) if table_htmls else None

            page_results.append(
                PageResult(
                    page_number=page_num,
                    text=markdown_text,
                    engine="ppstructure",
                    confidence=round(confidence, 4),
                    text_blocks=text_blocks,
                    layout_html=layout_html,
                )
            )

        return page_results

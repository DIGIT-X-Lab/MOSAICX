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
import re
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


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML ``<table>`` to a Markdown pipe table.

    PPStructureV3 renders detected tables as raw HTML.  LLMs and the CLI
    display work much better with Markdown ``| col | col |`` format.
    """
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.DOTALL)
    if not rows:
        return html
    md_rows: list[str] = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        cells = [c.strip() for c in cells]
        md_rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(md_rows)


def _reconstruct_text_block(
    block_bbox: list[int],
    rec_texts: list[str],
    dt_polys: list[list[list[float]]],
) -> str:
    """Reconstruct a text block's content using OCR y-coordinates.

    PPStructureV3 concatenates OCR lines within a ``text`` region into a
    single paragraph, losing line breaks.  This function finds all OCR
    detections that fall inside the block's bounding box, groups them by
    y-coordinate (same row), and joins with proper newlines.
    """
    bx0, by0, bx1, by1 = block_bbox
    # Collect OCR entries inside this block's bbox
    entries: list[tuple[float, float, str]] = []
    for text, poly in zip(rec_texts, dt_polys):
        ys = [p[1] for p in poly]
        xs = [p[0] for p in poly]
        cy = sum(ys) / len(ys)
        cx = sum(xs) / len(xs)
        # Check if center falls within the block bbox (with small margin)
        margin = 10
        if by0 - margin <= cy <= by1 + margin and bx0 - margin <= cx <= bx1 + margin:
            entries.append((cy, cx, text))

    if not entries:
        return ""

    # Group by y-coordinate into rows (entries within half median height
    # are on the same row)
    entries.sort(key=lambda e: (e[0], e[1]))
    rows: list[list[tuple[float, float, str]]] = [[entries[0]]]
    for entry in entries[1:]:
        prev_y = rows[-1][0][0]
        if abs(entry[0] - prev_y) < 20:  # same row threshold
            rows[-1].append(entry)
        else:
            rows.append([entry])

    # Sort entries within each row by x-position, join with space
    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda e: e[1])
        lines.append(" ".join(e[2] for e in row))

    return "\n".join(lines)


_HEADING_LABELS = frozenset({"doc_title", "paragraph_title"})
_TEXT_LABELS = frozenset({
    "text", "abstract", "content", "reference", "reference_content",
    "algorithm", "aside_text",
})


def _build_page_text(
    parsing_res: list,
    rec_texts: list[str],
    dt_polys: list[list[list[float]]],
) -> str:
    """Build page text from layout blocks with proper structure.

    Uses PPStructureV3's layout blocks for structure (headings, tables)
    and OCR y-coordinates for line breaks within text regions.
    """
    parts: list[str] = []

    for block in parsing_res:
        label = getattr(block, "label", None)
        if label is None and isinstance(block, dict):
            label = block.get("label")
        content = getattr(block, "content", None)
        if content is None and isinstance(block, dict):
            content = block.get("content")
        bbox = getattr(block, "bbox", None)
        if bbox is None and isinstance(block, dict):
            bbox = block.get("bbox")

        if not content and not bbox:
            continue

        if label == "table":
            # Convert HTML table to Markdown pipe table
            parts.append(_html_table_to_markdown(content or ""))
        elif label == "doc_title":
            text = (content or "").strip()
            # Strip <div> wrappers if present
            text = re.sub(r"<div[^>]*>(.*?)</div>", r"\1", text, flags=re.DOTALL)
            parts.append(f"# {text}")
        elif label in _HEADING_LABELS:
            text = (content or "").strip()
            text = re.sub(r"<div[^>]*>(.*?)</div>", r"\1", text, flags=re.DOTALL)
            parts.append(f"## {text}")
        elif label == "figure_title" or label == "chart_title":
            text = (content or "").strip()
            text = re.sub(r"<div[^>]*>(.*?)</div>", r"\1", text, flags=re.DOTALL)
            parts.append(text)
        elif label in _TEXT_LABELS and bbox is not None:
            # Reconstruct with proper line breaks from OCR coordinates
            reconstructed = _reconstruct_text_block(
                list(bbox), rec_texts, dt_polys
            )
            parts.append(reconstructed if reconstructed else (content or "").strip())
        elif content:
            # Fallback for any other label
            text = (content or "").strip()
            text = re.sub(r"<div[^>]*>(.*?)</div>", r"\1", text, flags=re.DOTALL)
            if text:
                parts.append(text)

    result = "\n\n".join(p for p in parts if p)
    # Collapse 3+ blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


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

        # Do NOT redirect stdout/stderr during predict — Rich's spinner
        # needs stdout to animate.  PaddlePaddle's logging is already
        # suppressed via logging.getLogger() in _get_ppstructure().
        results = list(engine.predict(str(path)))

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
        for i, page_result in enumerate(results):
            page_index = page_result["page_index"]
            page_num = (page_index + 1) if page_index is not None else (i + 1)

            # Extract raw OCR data
            ocr_res = page_result["overall_ocr_res"]
            if isinstance(ocr_res, dict):
                rec_texts = ocr_res.get("rec_texts", [])
                rec_scores = ocr_res.get("rec_scores", [])
                dt_polys = ocr_res.get("dt_polys", [])
            else:
                rec_texts = getattr(ocr_res, "rec_texts", [])
                rec_scores = getattr(ocr_res, "rec_scores", [])
                dt_polys = getattr(ocr_res, "dt_polys", [])

            # Reconstruct page text from layout blocks + OCR coordinates.
            # This preserves headings, tables as Markdown, and line breaks
            # within text regions (PPStructure's markdown_texts loses them).
            parsing_res = page_result["parsing_res_list"]
            page_text = _build_page_text(parsing_res, rec_texts, dt_polys)

            # Build TextBlocks with offsets into the reconstructed text
            text_blocks = _map_ocr_blocks_to_markdown(
                page_text,
                rec_texts,
                dt_polys,
                page_num=page_num,
                global_offset=0,
            )

            # Compute average confidence
            confidence = (
                sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
            )

            # Collect HTML from table blocks for layout_html field
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
                    text=page_text,
                    engine="ppstructure",
                    confidence=round(confidence, 4),
                    text_blocks=text_blocks,
                    layout_html=layout_html,
                )
            )

        return page_results

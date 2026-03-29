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


def _pages_from_raw(raw_pages: list[dict]) -> list[PageResult]:
    """Convert deserialized subprocess output into PageResult objects."""
    page_results: list[PageResult] = []

    for i, rp in enumerate(raw_pages):
        page_num = rp.get("page_num") or (i + 1)
        rec_texts = rp.get("rec_texts", [])
        rec_scores = rp.get("rec_scores", [])
        dt_polys = rp.get("dt_polys", [])
        blocks = rp.get("blocks", [])

        page_text = _build_page_text(blocks, rec_texts, dt_polys)

        text_blocks = _map_ocr_blocks_to_markdown(
            page_text, rec_texts, dt_polys,
            page_num=page_num, global_offset=0,
        )

        confidence = (
            sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
        )

        table_htmls = [
            b["content"] for b in blocks
            if b.get("label") == "table" and b.get("content")
        ]
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


# ---------------------------------------------------------------------------
# Subprocess script — runs PPStructureV3 in an isolated process so
# PaddlePaddle's GIL-holding C++ inference doesn't block the Rich spinner.
# Communicates via JSON on stdin/stdout.
# ---------------------------------------------------------------------------

_PPSTRUCTURE_SCRIPT = r"""
import io, json, logging, os, sys, warnings
import numpy as np

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
warnings.filterwarnings("ignore")
for _n in ("paddle", "paddleocr", "paddlex", "ppocr"):
    logging.getLogger(_n).setLevel(logging.WARNING)

data = json.loads(sys.stdin.read())
file_path = data["file_path"]
lang = data.get("lang")

# Suppress init noise
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

results = list(engine.predict(file_path))

# Serialize results — convert numpy types to Python native
pages = []
for i, r in enumerate(results):
    page_index = r["page_index"]
    page_num = (int(page_index) + 1) if page_index is not None else (i + 1)

    ocr_res = r["overall_ocr_res"]
    rec_texts = list(ocr_res.get("rec_texts", []) if isinstance(ocr_res, dict)
                     else getattr(ocr_res, "rec_texts", []))
    rec_scores = [float(s) for s in (
        ocr_res.get("rec_scores", []) if isinstance(ocr_res, dict)
        else getattr(ocr_res, "rec_scores", [])
    )]
    raw_polys = (ocr_res.get("dt_polys", []) if isinstance(ocr_res, dict)
                 else getattr(ocr_res, "dt_polys", []))
    dt_polys = []
    for poly in raw_polys:
        if isinstance(poly, np.ndarray):
            dt_polys.append(poly.tolist())
        else:
            dt_polys.append([[float(p[0]), float(p[1])] for p in poly])

    blocks = []
    for block in r["parsing_res_list"]:
        label = getattr(block, "label", None)
        if label is None and isinstance(block, dict):
            label = block.get("label")
        content = getattr(block, "content", None)
        if content is None and isinstance(block, dict):
            content = block.get("content")
        bbox = getattr(block, "bbox", None)
        if bbox is None and isinstance(block, dict):
            bbox = block.get("bbox")
        if bbox is not None:
            bbox = [int(x) for x in bbox]
        blocks.append({"label": label, "content": content, "bbox": bbox})

    pages.append({
        "page_num": page_num,
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "dt_polys": dt_polys,
        "blocks": blocks,
    })

# Write result to stdout (restored)
sys.stdout = _out
sys.stderr = _err
json.dump(pages, sys.stdout)
"""


class PPStructureEngine:
    """Layout-aware OCR engine backed by PaddleOCR PPStructureV3.

    Runs inference in a **subprocess** so PaddlePaddle's GIL-holding C++
    code doesn't block the Rich CLI spinner.  Falls back to in-process
    execution if the subprocess fails.
    """

    def __init__(self, lang: str | None = None):
        self._lang = lang

    def process_file(self, path: Path) -> list[PageResult]:
        """Process a document file and return one PageResult per page."""
        try:
            return self._run_subprocess(path)
        except Exception:
            logger.warning(
                "PPStructure subprocess failed, trying in-process",
                exc_info=True,
            )
            return self._run_in_process(path)

    def _run_subprocess(self, path: Path) -> list[PageResult]:
        """Run PPStructureV3 in a subprocess (spinner-friendly)."""
        import json
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", _PPSTRUCTURE_SCRIPT],
            input=json.dumps({
                "file_path": str(path),
                "lang": self._lang,
            }),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"PPStructure subprocess failed (exit {result.returncode}): "
                f"{result.stderr.strip()[-500:]}"
            )

        raw_pages = json.loads(result.stdout)
        if not raw_pages:
            return [
                PageResult(
                    page_number=1, text="", engine="ppstructure",
                    confidence=0.0,
                )
            ]

        return _pages_from_raw(raw_pages)

    def _run_in_process(self, path: Path) -> list[PageResult]:
        """Fallback: run PPStructureV3 in the current process."""
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        warnings.filterwarnings("ignore")
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
                lang=self._lang,
            )
        finally:
            sys.stdout, sys.stderr = _out, _err

        results = list(engine.predict(str(path)))
        if not results:
            return [
                PageResult(
                    page_number=1, text="", engine="ppstructure",
                    confidence=0.0,
                )
            ]

        # Convert to raw dicts and reuse shared builder
        import numpy as np

        raw_pages = []
        for i, r in enumerate(results):
            page_index = r["page_index"]
            page_num = (int(page_index) + 1) if page_index is not None else (i + 1)

            ocr_res = r["overall_ocr_res"]
            if isinstance(ocr_res, dict):
                rec_texts = list(ocr_res.get("rec_texts", []))
                rec_scores = [float(s) for s in ocr_res.get("rec_scores", [])]
                raw_polys = ocr_res.get("dt_polys", [])
            else:
                rec_texts = list(getattr(ocr_res, "rec_texts", []))
                rec_scores = [float(s) for s in getattr(ocr_res, "rec_scores", [])]
                raw_polys = getattr(ocr_res, "dt_polys", [])

            dt_polys = []
            for poly in raw_polys:
                if isinstance(poly, np.ndarray):
                    dt_polys.append(poly.tolist())
                else:
                    dt_polys.append([[float(p[0]), float(p[1])] for p in poly])

            blocks = []
            for block in r["parsing_res_list"]:
                label = getattr(block, "label", None)
                if label is None and isinstance(block, dict):
                    label = block.get("label")
                content = getattr(block, "content", None)
                if content is None and isinstance(block, dict):
                    content = block.get("content")
                bbox = getattr(block, "bbox", None)
                if bbox is None and isinstance(block, dict):
                    bbox = block.get("bbox")
                if bbox is not None:
                    bbox = [int(x) for x in bbox]
                blocks.append({"label": label, "content": content, "bbox": bbox})

            raw_pages.append({
                "page_num": page_num,
                "rec_texts": rec_texts,
                "rec_scores": rec_scores,
                "dt_polys": dt_polys,
                "blocks": blocks,
            })

        return _pages_from_raw(raw_pages)

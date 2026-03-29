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


def _transform_polys_to_pdf(
    dt_polys: list,
    img_w: int,
    img_h: int,
    pdf_w: float,
    pdf_h: float,
) -> list:
    """Transform OCR polygon coordinates from image pixels to PDF points.

    PPStructure coordinates are in image pixel space (origin top-left).
    PDF coordinates are in points (origin bottom-left, 72 DPI).
    """
    scale_x = pdf_w / img_w
    scale_y = pdf_h / img_h
    transformed = []
    for poly in dt_polys:
        new_poly = []
        for x, y in poly:
            # Scale from pixels to points, flip Y axis
            px = float(x) * scale_x
            py = pdf_h - float(y) * scale_y
            new_poly.append([px, py])
        transformed.append(new_poly)
    return transformed


def _pages_from_raw(
    raw_pages: list[dict],
    pdf_page_dims: list[tuple[float, float]] | None = None,
) -> list[PageResult]:
    """Convert deserialized subprocess output into PageResult objects.

    Parameters
    ----------
    pdf_page_dims:
        If the source is a PDF, provide ``[(width, height), ...]`` in
        PDF points so TextBlock bboxes can be transformed from image
        pixel coordinates to PDF coordinates for accurate redaction.
    """
    page_results: list[PageResult] = []

    for i, rp in enumerate(raw_pages):
        page_num = rp.get("page_num") or (i + 1)
        rec_texts = rp.get("rec_texts", [])
        rec_scores = rp.get("rec_scores", [])
        dt_polys_orig = rp.get("dt_polys", [])
        blocks = rp.get("blocks", [])

        # Build page text using ORIGINAL pixel coordinates (same space
        # as block bboxes from parsing_res_list).
        page_text = _build_page_text(blocks, rec_texts, dt_polys_orig)

        # Transform coordinates from image pixels to PDF points for
        # TextBlocks (used by redaction/provenance).
        dt_polys_final = dt_polys_orig
        img_w = rp.get("img_width")
        img_h = rp.get("img_height")
        if pdf_page_dims and img_w and img_h and i < len(pdf_page_dims):
            pdf_w, pdf_h = pdf_page_dims[i]
            dt_polys_final = _transform_polys_to_pdf(
                dt_polys_orig, img_w, img_h, pdf_w, pdf_h
            )

        text_blocks = _map_ocr_blocks_to_markdown(
            page_text, rec_texts, dt_polys_final,
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
# Persistent worker subprocess — PPStructureV3 runs in an isolated process
# so PaddlePaddle's GIL-holding C++ inference doesn't block the Rich
# spinner.  The worker stays alive between calls so the model loads once.
#
# Protocol: parent writes one JSON line to stdin, worker writes one JSON
# line to stdout per request.  Worker exits when stdin is closed.
# ---------------------------------------------------------------------------

_PPSTRUCTURE_WORKER_SCRIPT = r"""
import io, json, logging, os, sys, warnings

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
warnings.filterwarnings("ignore")
for _n in ("paddle", "paddleocr", "paddlex", "ppocr"):
    logging.getLogger(_n).setLevel(logging.WARNING)

import numpy as np

_engine = None
_engine_lang = None


def _get_engine(lang):
    global _engine, _engine_lang
    if _engine is not None and _engine_lang == lang:
        return _engine
    _out, _err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        from paddleocr import PPStructureV3
        _engine = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_table_recognition=True,
            use_region_detection=True,
            lang=lang,
        )
        _engine_lang = lang
    finally:
        sys.stdout, sys.stderr = _out, _err
    return _engine


def _serialize_results(results):
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

        # Image dimensions for coordinate transformation
        img_w = int(r["width"]) if r.get("width") else None
        img_h = int(r["height"]) if r.get("height") else None

        pages.append({
            "page_num": page_num,
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "dt_polys": dt_polys,
            "blocks": blocks,
            "img_width": img_w,
            "img_height": img_h,
        })
    return pages


# Main loop: read requests from stdin, write responses to stdout
_real_stdout = sys.stdout
_real_stderr = sys.stderr
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
        engine = _get_engine(data.get("lang"))
        results = list(engine.predict(data["file_path"]))
        response = {"ok": True, "pages": _serialize_results(results)}
    except Exception as e:
        response = {"ok": False, "error": str(e)}
    # Ensure we write to the real stdout (not any redirected one)
    sys.stdout = _real_stdout
    sys.stderr = _real_stderr
    _real_stdout.write(json.dumps(response) + "\n")
    _real_stdout.flush()
"""


# Module-level persistent worker process
_worker_proc: "subprocess.Popen | None" = None
_worker_lock: "threading.Lock | None" = None


def _get_worker() -> "subprocess.Popen":
    """Get or spawn the persistent PPStructure worker process."""
    import subprocess
    import threading

    global _worker_proc, _worker_lock

    if _worker_lock is None:
        _worker_lock = threading.Lock()

    with _worker_lock:
        if _worker_proc is not None and _worker_proc.poll() is None:
            return _worker_proc

        # Spawn a new worker
        _worker_proc = subprocess.Popen(
            [sys.executable, "-c", _PPSTRUCTURE_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        return _worker_proc


class PPStructureEngine:
    """Layout-aware OCR engine backed by PaddleOCR PPStructureV3.

    Runs inference in a **persistent worker subprocess** so PaddlePaddle's
    GIL-holding C++ code doesn't block the Rich CLI spinner.  The worker
    stays alive between calls — the model loads once on first use.
    """

    def __init__(self, lang: str | None = None):
        self._lang = lang

    def process_file(
        self,
        path: Path,
        pdf_page_dims: list[tuple[float, float]] | None = None,
    ) -> list[PageResult]:
        """Process a document file and return one PageResult per page.

        Parameters
        ----------
        pdf_page_dims:
            For PDF files, provide page dimensions in PDF points so
            TextBlock bboxes can be transformed from image pixels to
            PDF coordinates for accurate redaction.
        """
        self._pdf_page_dims = pdf_page_dims
        try:
            return self._run_worker(path)
        except Exception:
            logger.warning(
                "PPStructure worker failed, trying fresh subprocess",
                exc_info=True,
            )
            # Kill stale worker so next call spawns a fresh one
            global _worker_proc
            if _worker_proc is not None:
                try:
                    _worker_proc.kill()
                except OSError:
                    pass
                _worker_proc = None
            return self._run_oneshot_subprocess(path)

    def _run_worker(self, path: Path) -> list[PageResult]:
        """Send a request to the persistent worker subprocess."""
        import json

        worker = _get_worker()
        request = json.dumps({"file_path": str(path), "lang": self._lang})
        worker.stdin.write(request + "\n")
        worker.stdin.flush()

        # Read exactly one response line (blocks until worker responds,
        # but releases the GIL so Rich spinner can animate)
        response_line = worker.stdout.readline()
        if not response_line:
            raise RuntimeError("Worker process died (no response)")

        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(
                f"Worker error: {response.get('error', 'unknown')}"
            )

        raw_pages = response.get("pages", [])
        if not raw_pages:
            return [
                PageResult(
                    page_number=1, text="", engine="ppstructure",
                    confidence=0.0,
                )
            ]

        return _pages_from_raw(raw_pages, pdf_page_dims=self._pdf_page_dims)

    def _run_oneshot_subprocess(self, path: Path) -> list[PageResult]:
        """Fallback: one-shot subprocess (re-inits model, slower)."""
        import json
        import subprocess

        # Reuse the worker script — send one request then close stdin
        proc = subprocess.Popen(
            [sys.executable, "-c", _PPSTRUCTURE_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        request = json.dumps({"file_path": str(path), "lang": self._lang})
        stdout, stderr = proc.communicate(input=request + "\n", timeout=300)

        if proc.returncode != 0:
            raise RuntimeError(
                f"PPStructure subprocess failed: {stderr.strip()[-500:]}"
            )

        # Parse the first JSON line from stdout
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(
                    f"Subprocess error: {response.get('error', 'unknown')}"
                )
            raw_pages = response.get("pages", [])
            if not raw_pages:
                return [
                    PageResult(
                        page_number=1, text="", engine="ppstructure",
                        confidence=0.0,
                    )
                ]
            return _pages_from_raw(raw_pages, pdf_page_dims=self._pdf_page_dims)

        raise RuntimeError("No output from PPStructure subprocess")

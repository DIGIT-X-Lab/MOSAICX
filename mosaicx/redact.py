# mosaicx/redact.py
"""Produce redacted documents from deidentification results.

Supports multiple output formats:
- PDF: PyMuPDF redaction annotations (text physically removed)
- Images (PNG/JPG/TIFF): PIL black rectangles over PHI regions
- Text (TXT/MD): direct string replacement
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REDACTION_COLOR = (0, 0, 0)  # uniform black


def create_redacted_document(
    source_path: Path,
    output_path: Path,
    redacted_text: str,
    redaction_map: list[dict[str, Any]],
    page_dimensions: list[tuple[float, float]] | None = None,
    loaded_doc: Any = None,
) -> Path:
    """Create a redacted copy of a document in its original format.

    Dispatches to format-specific redaction based on the source file
    extension.

    Parameters
    ----------
    source_path:
        Path to the original document.
    output_path:
        Where to save the redacted document.
    redacted_text:
        The redacted text (used for text file output).
    redaction_map:
        Enriched redaction map entries (with optional ``spans``
        containing ``page`` and ``bbox`` from provenance).
    page_dimensions:
        Optional list of ``(width, height)`` per page.
    loaded_doc:
        Optional LoadedDocument with OCR metadata (preprocessed image).

    Returns
    -------
    Path
        The path to the saved redacted document.
    """
    suffix = source_path.suffix.lower()

    if suffix == ".pdf":
        return _redact_pdf(source_path, output_path, redaction_map,
                           page_dimensions)
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
        return _redact_image(source_path, output_path, redaction_map,
                             loaded_doc)
    elif suffix in (".txt", ".md"):
        return _redact_text(output_path, redacted_text)
    else:
        raise ValueError(
            f"Redacted output not supported for format '{suffix}'. "
            f"Supported: .pdf, .png, .jpg, .tiff, .txt, .md"
        )


# Keep backward-compatible alias
create_redacted_pdf = None  # replaced below after _redact_pdf definition


def _redact_pdf(
    source_pdf: Path,
    output_path: Path,
    redaction_map: list[dict[str, Any]],
    page_dimensions: list[tuple[float, float]] | None = None,
) -> Path:
    """Create a redacted PDF with PHI physically removed."""
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF required for PDF redaction: pip install pymupdf"
        )

    doc = fitz.open(str(source_pdf))

    for entry in redaction_map:
        original = entry.get("original", "")
        replacement = entry.get("replacement", "[REDACTED]")
        spans = entry.get("spans", [])

        if spans:
            for span in spans:
                page_num = span["page"] - 1
                if page_num < 0 or page_num >= len(doc):
                    continue
                page = doc[page_num]
                bbox = span["bbox"]
                x0, y0_pdf, x1, y1_pdf = bbox

                page_h = page.rect.height
                rect = fitz.Rect(x0, page_h - y1_pdf, x1, page_h - y0_pdf)
                rect = rect + fitz.Rect(-1, -1, 1, 1)

                page.add_redact_annot(
                    rect,
                    text=replacement,
                    fontsize=8,
                    fill=_REDACTION_COLOR,
                    text_color=(1, 1, 1),
                )
        else:
            search_text = original.strip()
            if not search_text:
                continue
            for page in doc:
                rects = page.search_for(search_text)
                for rect in rects:
                    page.add_redact_annot(
                        rect,
                        text=replacement,
                        fontsize=8,
                        fill=_REDACTION_COLOR,
                        text_color=(1, 1, 1),
                    )

    for page in doc:
        page.apply_redactions()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


# Backward-compatible alias for cli.py import
create_redacted_pdf = _redact_pdf


def _redact_image(
    source_image: Path,
    output_path: Path,
    redaction_map: list[dict[str, Any]],
    loaded_doc: Any = None,
) -> Path:
    """Create a redacted image with black rectangles over PHI regions.

    PaddleOCR internally preprocesses images (resize, unwarp) before
    detection, so its bounding box coordinates are in the preprocessed
    image's coordinate space — not the original.

    Strategy: draw redaction rectangles on the preprocessed image
    (where coordinates are guaranteed correct), then resize back to
    original dimensions.  The preprocessed image is captured during
    document loading and stored on the LoadedDocument.
    """
    from PIL import Image, ImageDraw

    orig_img = Image.open(source_image).convert("RGB")
    orig_size = orig_img.size

    phi_texts = [
        entry.get("original", "").strip()
        for entry in redaction_map
        if entry.get("original", "").strip()
    ]

    if not phi_texts:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        orig_img.save(str(output_path))
        return output_path

    # Run a SINGLE OCR pass to get coordinates and the preprocessed
    # image dimensions. PaddleOCR's dt_polys are in the preprocessed
    # image's coordinate space (which may differ from the original by
    # padding/resizing). We scale coordinates back to the original
    # image and draw directly on it.
    ocr_blocks, ocr_pre_img = _single_ocr_pass(source_image)

    if ocr_blocks:
        draw = ImageDraw.Draw(orig_img)

        # Compute scale factor: OCR coords are in preprocessed space
        # (panel 0 of the triptych), original image may differ in size.
        if ocr_pre_img is not None:
            if hasattr(ocr_pre_img, "size"):
                pre_w, pre_h = ocr_pre_img.size
            else:
                pre_h, pre_w = ocr_pre_img.shape[:2]
            # Panel 0 width = total width / 3
            panel_w = pre_w // 3
            panel_h = pre_h
        else:
            panel_w, panel_h = orig_size

        sx = orig_size[0] / panel_w if panel_w > 0 else 1.0
        sy = orig_size[1] / panel_h if panel_h > 0 else 1.0

        for block_text, block_bbox in ocr_blocks:
            bt = block_text.strip()
            for phi in phi_texts:
                if phi in bt or bt in phi:
                    x0, y0, x1, y1 = block_bbox
                    # Scale from OCR space to original image space
                    rx0 = x0 * sx - 3
                    ry0 = y0 * sy - 3
                    rx1 = x1 * sx + 3
                    ry1 = y1 * sy + 3
                    draw.rectangle(
                        [rx0, ry0, rx1, ry1],
                        fill=_REDACTION_COLOR,
                    )
                    break

        result = orig_img
    else:
        logger.warning(
            "OCR unavailable for image redaction — cannot redact."
        )
        result = orig_img

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_suffix = output_path.suffix.lower()
    fmt_map = {
        ".jpg": "JPEG", ".jpeg": "JPEG",
        ".png": "PNG",
        ".tiff": "TIFF", ".tif": "TIFF",
        ".bmp": "BMP",
    }
    img_format = fmt_map.get(out_suffix, "PNG")
    result.save(str(output_path), format=img_format)
    return output_path


def _single_ocr_pass(
    image_path: Path,
) -> tuple[list[tuple[str, tuple[float, float, float, float]]], Any]:
    """Run PaddleOCR once and return blocks + preprocessed image together.

    This ensures coordinates and image come from the SAME OCR run,
    avoiding non-determinism issues across separate runs.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.warning("PaddleOCR not available for image redaction")
        return [], None

    import io
    import os
    import sys

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    _out, _err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        ocr = PaddleOCR(lang="german")
        results = list(ocr.predict(str(image_path)))
    finally:
        sys.stdout, sys.stderr = _out, _err

    if not results:
        return [], None

    page_data = results[0]
    res = page_data.json.get("res", {})
    rec_texts = res.get("rec_texts", [])
    dt_polys = res.get("dt_polys", [])

    blocks: list[tuple[str, tuple[float, float, float, float]]] = []
    for i in range(min(len(rec_texts), len(dt_polys))):
        poly = dt_polys[i]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        blocks.append((rec_texts[i], (min(xs), min(ys), max(xs), max(ys))))

    pre_img = None
    img_dict = getattr(page_data, "img", None)
    if isinstance(img_dict, dict):
        pre_img = img_dict.get("preprocessed_img")

    return blocks, pre_img


def _redact_text(
    output_path: Path,
    redacted_text: str,
) -> Path:
    """Write the redacted text to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(redacted_text, encoding="utf-8")
    return output_path

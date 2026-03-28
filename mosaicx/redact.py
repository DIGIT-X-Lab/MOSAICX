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
        return _redact_image(source_path, output_path, redaction_map)
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
) -> Path:
    """Create a redacted image with black rectangles over PHI regions."""
    from PIL import Image, ImageDraw

    img = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(img)

    for entry in redaction_map:
        original = entry.get("original", "")
        spans = entry.get("spans", [])

        if spans:
            for span in spans:
                bbox = span["bbox"]
                x0, y0, x1, y1 = bbox
                # OCR bboxes are in image pixel space (top-left origin)
                draw.rectangle([x0, y0, x1, y1], fill=_REDACTION_COLOR)
        else:
            if original:
                logger.warning(
                    "No bounding box for PHI '%s' — cannot redact on image. "
                    "Ensure OCR engine provides coordinates.",
                    original[:30],
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_suffix = output_path.suffix.lower()
    fmt_map = {
        ".jpg": "JPEG", ".jpeg": "JPEG",
        ".png": "PNG",
        ".tiff": "TIFF", ".tif": "TIFF",
        ".bmp": "BMP",
    }
    img_format = fmt_map.get(out_suffix, "PNG")
    img.save(str(output_path), format=img_format)
    return output_path


def _redact_text(
    output_path: Path,
    redacted_text: str,
) -> Path:
    """Write the redacted text to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(redacted_text, encoding="utf-8")
    return output_path

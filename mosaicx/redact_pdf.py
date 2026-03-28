# mosaicx/redact_pdf.py
"""Produce a redacted PDF from deidentification results.

Uses PyMuPDF (fitz) to apply true PDF redactions — the original PHI text
is physically removed from the document, not merely covered.  Each
redaction is color-coded by PHI type.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Uniform redaction color — black bars, no PHI type leakage
_REDACTION_COLOR = (0.0, 0.0, 0.0)


def create_redacted_pdf(
    source_pdf: Path,
    output_path: Path,
    redaction_map: list[dict[str, Any]],
    page_dimensions: list[tuple[float, float]] | None = None,
) -> Path:
    """Create a redacted copy of a PDF with PHI physically removed.

    Parameters
    ----------
    source_pdf:
        Path to the original PDF document.
    output_path:
        Where to save the redacted PDF.
    redaction_map:
        Enriched redaction map entries (with ``spans`` containing
        ``page`` and ``bbox`` from provenance).  Entries without
        ``spans`` are handled via text search fallback.
    page_dimensions:
        Optional list of ``(width, height)`` per page in PDF points.
        Used for coordinate system conversion if needed.

    Returns
    -------
    Path
        The path to the saved redacted PDF.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF required for PDF redaction: pip install pymupdf"
        )

    doc = fitz.open(str(source_pdf))

    for entry in redaction_map:
        original = entry.get("original", "")
        phi_type = entry.get("phi_type", "OTHER")
        replacement = entry.get("replacement", "[REDACTED]")
        color = _REDACTION_COLOR
        spans = entry.get("spans", [])

        if spans:
            # Use provenance coordinates
            for span in spans:
                page_num = span["page"] - 1  # 0-indexed
                if page_num < 0 or page_num >= len(doc):
                    continue
                page = doc[page_num]
                bbox = span["bbox"]
                x0, y0_pdf, x1, y1_pdf = bbox

                # PDF coords: origin bottom-left, y up.
                # PyMuPDF uses top-left origin. Convert.
                page_h = page.rect.height
                rect = fitz.Rect(x0, page_h - y1_pdf, x1, page_h - y0_pdf)

                # Expand rect slightly for visual clarity
                rect = rect + fitz.Rect(-1, -1, 1, 1)

                page.add_redact_annot(
                    rect,
                    text=replacement,
                    fontsize=8,
                    fill=color,
                    text_color=(1, 1, 1),
                )
        else:
            # Fallback: search for the original text in the PDF
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
                        fill=color,
                        text_color=(1, 1, 1),
                    )

    # Apply all redactions — this physically removes the text
    for page in doc:
        page.apply_redactions()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()

    return output_path

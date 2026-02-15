# mosaicx/documents/engines/base.py
"""OCR engine protocol and shared utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image

from ..models import PageResult

# All formats the loader accepts
TEXT_FORMATS = frozenset({".txt", ".md", ".markdown"})
IMAGE_FORMATS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"})
PDF_FORMATS = frozenset({".pdf"})
DOCLING_FORMATS = frozenset({".docx", ".pptx"})  # still text-extractable
SUPPORTED_FORMATS = TEXT_FORMATS | IMAGE_FORMATS | PDF_FORMATS | DOCLING_FORMATS


@runtime_checkable
class OCREngine(Protocol):
    """Protocol for OCR engines."""

    def ocr_pages(self, images: list[Image.Image], langs: list[str] | None = None) -> list[PageResult]:
        """Run OCR on a list of page images.

        Parameters
        ----------
        images : list of PIL Images, one per page.
        langs  : language hints (e.g. ["en", "de"]).

        Returns
        -------
        list[PageResult] with one entry per input image.
        """
        ...


def pdf_to_images(path: Path, dpi: int = 200) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Images (one per page).

    Uses pypdfium2 — no system dependencies (unlike poppler).
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=dpi / 72)
        images.append(bitmap.to_pil())
    pdf.close()
    return images


def image_to_pages(path: Path) -> list[Image.Image]:
    """Load an image file as a single-page list."""
    img = Image.open(path).convert("RGB")
    return [img]

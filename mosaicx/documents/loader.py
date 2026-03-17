# mosaicx/documents/loader.py
"""Document loading orchestrator — OCR via PaddleOCR or Chandra.

Routes documents through the appropriate loading path:
- .txt/.md: direct read (no OCR)
- .pdf/.jpg/.png/.tiff: OCR with PaddleOCR (default) or Chandra
- .docx/.pptx: text extraction (if docling available), else error
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .engines.base import (
    IMAGE_FORMATS,
    PDF_FORMATS,
    SUPPORTED_FORMATS,
    TEXT_FORMATS,
    image_to_pages,
    pdf_to_images,
)
from .models import DocumentLoadError, LoadedDocument, PageResult
from .quality import QualityScorer

logger = logging.getLogger(__name__)

_scorer = QualityScorer()


def load_document(
    path: Path,
    ocr_engine: str = "paddleocr",
    force_ocr: bool = False,
    ocr_langs: list[str] | None = None,
    chandra_backend: str | None = None,
    quality_threshold: float = 0.6,
    page_timeout: int = 60,
) -> LoadedDocument:
    """Load a document from disk with automatic OCR if needed.

    Parameters
    ----------
    path            : Path to the document.
    ocr_engine      : "paddleocr" or "chandra".
    force_ocr       : Run OCR even on native PDFs with text layers.
    ocr_langs       : Language hints (e.g. ["en", "de"]).
    chandra_backend : "vllm", "hf", or None for auto-detect.
    quality_threshold : Minimum quality score before flagging warning.
    page_timeout    : Max seconds per page for each engine.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    # Text files: direct read, no OCR
    if suffix in TEXT_FORMATS:
        return _load_text(path, suffix.lstrip("."))

    # PDF: try native text extraction first, fall back to OCR
    if suffix in PDF_FORMATS:
        if not force_ocr:
            native = _try_native_pdf_text(path)
            if native is not None:
                return native
        images = pdf_to_images(path)
    elif suffix in IMAGE_FORMATS:
        images = image_to_pages(path)
    else:
        raise DocumentLoadError(f"Cannot process format: {suffix}")

    if not images:
        return LoadedDocument(
            text="", source_path=path, format=suffix.lstrip("."),
            page_count=0, quality_warning=True,
        )

    # Run OCR engine (single engine, main thread — avoids MPS threading issues)
    pages = _run_engine(
        images=images,
        ocr_engine=ocr_engine,
        ocr_langs=ocr_langs or ["en"],
        chandra_backend=chandra_backend,
    )

    return _assemble_document(
        winners=pages,
        source_path=path,
        fmt=suffix.lstrip("."),
        threshold=quality_threshold,
    )


def _load_text(path: Path, fmt: str) -> LoadedDocument:
    """Load a plain text file."""
    text = path.read_text(encoding="utf-8")
    return LoadedDocument(text=text, source_path=path, format=fmt)


def _try_native_pdf_text(path: Path) -> Optional[LoadedDocument]:
    """Try to extract text from a PDF's native text layer.

    Returns a LoadedDocument if the PDF has a usable text layer (>50 chars),
    or None if OCR is needed.
    """
    try:
        import pypdfium2
    except ImportError:
        return None

    try:
        pdf = pypdfium2.PdfDocument(str(path))
        pages_text: list[str] = []
        for page in pdf:
            tp = page.get_textpage()
            text = tp.get_text_bounded()
            pages_text.append(text)
            tp.close()
            page.close()
        pdf.close()

        full_text = "\n\n".join(pages_text)
        # If native text is too sparse, fall back to OCR
        if len(full_text.strip()) < 50:
            return None

        return LoadedDocument(
            text=full_text,
            source_path=path,
            format="pdf",
            page_count=len(pages_text),
        )
    except Exception:
        logger.debug("Native PDF text extraction failed for %s", path)
        return None


def _run_engine(
    images: list,
    ocr_engine: str,
    ocr_langs: list[str],
    chandra_backend: str | None,
) -> list[PageResult]:
    """Dispatch a single OCR engine in the main thread and return results."""
    if ocr_engine == "chandra":
        try:
            from .engines.chandra_engine import ChandraEngine
            engine = ChandraEngine(backend=chandra_backend)
            return engine.ocr_pages(images)
        except Exception:
            logger.warning("OCR engine 'chandra' failed, skipping", exc_info=True)
            return [
                PageResult(page_number=i + 1, text="", engine="chandra", confidence=0.0)
                for i in range(len(images))
            ]

    # Default: paddleocr
    try:
        from .engines.paddleocr_engine import PaddleOCREngine
        engine = PaddleOCREngine()
        return engine.ocr_pages(images, langs=ocr_langs)
    except Exception:
        logger.warning("OCR engine 'paddleocr' failed, skipping", exc_info=True)
        return [
            PageResult(page_number=i + 1, text="", engine="paddleocr", confidence=0.0)
            for i in range(len(images))
        ]


def _assemble_document(
    winners: list[PageResult],
    source_path: Path,
    fmt: str,
    threshold: float = 0.6,
) -> LoadedDocument:
    """Assemble a LoadedDocument from per-page results."""
    if not winners:
        return LoadedDocument(
            text="", source_path=source_path, format=fmt,
            page_count=0, quality_warning=True,
        )

    full_text = "\n\n".join(p.text for p in winners if p.text)
    engines_used = set(p.engine for p in winners if p.text)
    avg_confidence = (
        sum(p.confidence for p in winners) / len(winners) if winners else 0.0
    )
    any_below = any(p.confidence < threshold for p in winners)

    if engines_used:
        engine_label = engines_used.pop()
    else:
        engine_label = None

    return LoadedDocument(
        text=full_text,
        source_path=source_path,
        format=fmt,
        page_count=len(winners),
        ocr_engine_used=engine_label,
        ocr_confidence=round(avg_confidence, 3),
        quality_warning=any_below,
        pages=winners,
    )

# mosaicx/documents/loader.py
"""Document loading orchestrator — dual-engine OCR (Surya + Chandra).

Routes documents through the appropriate loading path:
- .txt/.md: direct read (no OCR)
- .pdf/.jpg/.png/.tiff: parallel OCR with Surya + Chandra, quality scoring
- .docx/.pptx: text extraction (if docling available), else error
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ocr_engine: str = "both",
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
    ocr_engine      : "both", "surya", or "chandra".
    force_ocr       : Run OCR even on native PDFs with text layers.
    ocr_langs       : Language hints for Surya (e.g. ["en", "de"]).
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

    # Run OCR engines
    surya_pages, chandra_pages = _run_engines(
        images=images,
        ocr_engine=ocr_engine,
        ocr_langs=ocr_langs or ["en"],
        chandra_backend=chandra_backend,
        page_timeout=page_timeout,
    )

    # Pick best per page
    winners = _pick_best_pages(surya_pages, chandra_pages, threshold=quality_threshold)

    return _assemble_document(
        winners=winners,
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


def _run_engines(
    images: list,
    ocr_engine: str,
    ocr_langs: list[str],
    chandra_backend: str | None,
    page_timeout: int,
) -> tuple[list[PageResult], list[PageResult]]:
    """Dispatch OCR engines in parallel and collect results."""
    surya_pages: list[PageResult] = []
    chandra_pages: list[PageResult] = []

    def run_surya():
        from .engines.surya_engine import SuryaEngine
        engine = SuryaEngine()
        return engine.ocr_pages(images, langs=ocr_langs)

    def run_chandra():
        from .engines.chandra_engine import ChandraEngine
        engine = ChandraEngine(backend=chandra_backend)
        return engine.ocr_pages(images)

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if ocr_engine in ("both", "surya"):
            futures["surya"] = pool.submit(run_surya)
        if ocr_engine in ("both", "chandra"):
            futures["chandra"] = pool.submit(run_chandra)

        for name, future in futures.items():
            try:
                max_timeout = min(page_timeout * len(images), 600)  # cap at 10 min
                result = future.result(timeout=max_timeout)
            except Exception:
                logger.warning("OCR engine '%s' failed, skipping", name)
                empty = [
                    PageResult(page_number=i + 1, text="", engine=name, confidence=0.0)
                    for i in range(len(images))
                ]
                result = empty

            if name == "surya":
                surya_pages = result
            else:
                chandra_pages = result

    return surya_pages, chandra_pages


def _pick_best_pages(
    surya_pages: list[PageResult],
    chandra_pages: list[PageResult],
    threshold: float = 0.6,
) -> list[PageResult]:
    """Compare per-page quality scores and pick the best engine for each page."""
    # Handle single-engine cases
    if not surya_pages and not chandra_pages:
        return []
    if not surya_pages:
        return chandra_pages
    if not chandra_pages:
        return surya_pages

    winners = []
    page_count = max(len(surya_pages), len(chandra_pages))

    for i in range(page_count):
        s_page = surya_pages[i] if i < len(surya_pages) else None
        c_page = chandra_pages[i] if i < len(chandra_pages) else None

        if s_page is None:
            winners.append(c_page)
            continue
        if c_page is None:
            winners.append(s_page)
            continue

        # Score both using the quality scorer
        s_score = _scorer.score(s_page.text)
        c_score = _scorer.score(c_page.text)

        # Update confidence with quality score
        s_page = PageResult(
            page_number=s_page.page_number, text=s_page.text,
            engine=s_page.engine, confidence=s_score,
            layout_html=s_page.layout_html,
        )
        c_page = PageResult(
            page_number=c_page.page_number, text=c_page.text,
            engine=c_page.engine, confidence=c_score,
            layout_html=c_page.layout_html,
        )

        winners.append(s_page if s_score >= c_score else c_page)

    return winners


def _assemble_document(
    winners: list[PageResult],
    source_path: Path,
    fmt: str,
    threshold: float = 0.6,
) -> LoadedDocument:
    """Assemble a LoadedDocument from per-page winners."""
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

    if len(engines_used) > 1:
        engine_label = "mixed"
    elif engines_used:
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

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
from .models import DocumentLoadError, LoadedDocument, PageResult, TextBlock
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
    or None if OCR is needed.  Also builds word-level TextBlocks with bounding
    boxes from pypdfium2 character position data.
    """
    try:
        import pypdfium2
    except ImportError:
        return None

    try:
        pdf = pypdfium2.PdfDocument(str(path))
        pages_text: list[str] = []
        all_blocks: list[TextBlock] = []
        page_dimensions: list[tuple[float, float]] = []
        global_offset = 0

        for page_num_zero, page in enumerate(pdf):
            tp = page.get_textpage()
            page_text = tp.get_text_bounded()
            pages_text.append(page_text)

            # Collect page dimensions (width, height)
            page_dimensions.append((page.get_width(), page.get_height()))

            # Build word-level TextBlocks from character bounding boxes
            blocks = _build_textblocks_from_textpage(
                tp, page_text, page_num_zero + 1, global_offset
            )
            all_blocks.extend(blocks)

            tp.close()
            page.close()

            # Advance global offset: page text length + 2-char "\n\n" separator
            # (except after the last page where no separator is appended)
            global_offset += len(page_text) + 2  # +2 for the "\n\n" joiner

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
            text_blocks=all_blocks,
            page_dimensions=page_dimensions,
        )
    except Exception:
        logger.debug("Native PDF text extraction failed for %s", path)
        return None


def _flush_word(
    blocks: list[TextBlock],
    word_chars: list[tuple[float, float, float, float]],
    word_start: int,
    word_end: int,
    page_text: str,
    page_num: int,
    global_offset: int,
) -> None:
    """Flush accumulated word characters into a TextBlock and append to blocks."""
    if not word_chars:
        return
    x0 = min(box[0] for box in word_chars)
    y0 = min(box[1] for box in word_chars)
    x1 = max(box[2] for box in word_chars)
    y1 = max(box[3] for box in word_chars)
    text = page_text[word_start:word_end]
    abs_start = global_offset + word_start
    abs_end = global_offset + word_end
    blocks.append(
        TextBlock(
            text=text,
            start=abs_start,
            end=abs_end,
            page=page_num,
            bbox=(x0, y0, x1, y1),
        )
    )


def _build_textblocks_from_textpage(
    textpage: object,
    page_text: str,
    page_num: int,
    global_offset: int,
) -> list[TextBlock]:
    """Build word-level TextBlocks from pypdfium2 textpage character boxes.

    Iterates characters in the textpage, groups adjacent non-whitespace
    characters into words, and computes the union bounding box for each word.

    Parameters
    ----------
    textpage:
        A pypdfium2 PdfTextPage object.
    page_text:
        The full text string for this page (from ``get_text_bounded()``).
    page_num:
        1-indexed page number for the TextBlock records.
    global_offset:
        Character offset of the first character of this page in the
        assembled full-document text string.

    Returns
    -------
    list[TextBlock]
        Word-level TextBlocks sorted by start offset.
    """
    blocks: list[TextBlock] = []
    n_chars = len(page_text)

    # Accumulate non-whitespace characters into words
    word_chars: list[tuple[float, float, float, float]] = []
    word_start: int = 0  # page-local char index

    for i in range(n_chars):
        ch = page_text[i]
        if ch.isspace():
            # Flush current word if any
            if word_chars:
                _flush_word(
                    blocks, word_chars, word_start, i,
                    page_text, page_num, global_offset,
                )
                word_chars = []
            continue

        # Non-whitespace character — try to get its bounding box
        try:
            left, bottom, right, top = textpage.get_charbox(i)
            # Normalise to (x0, y0, x1, y1) with y0 <= y1
            x0 = min(left, right)
            x1 = max(left, right)
            y0 = min(bottom, top)
            y1 = max(bottom, top)
            # Skip degenerate boxes (zero-area — usually control chars / glyphs
            # without valid glyph data)
            if x0 == x1 and y0 == y1:
                # Still part of the word — start a new word after flushing if needed
                if not word_chars:
                    word_start = i
                # We cannot contribute a box but we keep the char in the word span;
                # just don't add a box entry
                if not word_chars and i > 0 and not page_text[i - 1].isspace():
                    pass  # continuing a word with no box — handled below
                # Track word boundaries even without a box
                if not word_chars:
                    word_start = i
                continue
        except Exception:
            # get_charbox can fail for some characters; skip the box
            if not word_chars:
                word_start = i
            continue

        # Valid box acquired
        if not word_chars:
            word_start = i
        word_chars.append((x0, y0, x1, y1))

    # Flush any remaining word at end of page
    if word_chars:
        _flush_word(
            blocks, word_chars, word_start, n_chars,
            page_text, page_num, global_offset,
        )

    return blocks


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
    """Assemble a LoadedDocument from per-page results.

    Also merges per-page TextBlocks (if any) into a single flat list,
    adjusting each block's start/end offsets to be relative to the full
    document text (pages joined with "\\n\\n" separators).
    """
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

    # Merge TextBlocks from each page, adjusting offsets for the "\n\n" separator
    all_blocks: list[TextBlock] = []
    global_offset = 0
    for page in winners:
        if page.text_blocks:
            for tb in page.text_blocks:
                all_blocks.append(
                    TextBlock(
                        text=tb.text,
                        start=tb.start + global_offset,
                        end=tb.end + global_offset,
                        page=tb.page,
                        bbox=tb.bbox,
                    )
                )
        if page.text:
            global_offset += len(page.text) + 2  # +2 for "\n\n" separator

    return LoadedDocument(
        text=full_text,
        source_path=source_path,
        format=fmt,
        page_count=len(winners),
        ocr_engine_used=engine_label,
        ocr_confidence=round(avg_confidence, 3),
        quality_warning=any_below,
        pages=winners,
        text_blocks=all_blocks,
    )

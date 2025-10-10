"""
Unified document loading utilities for MOSAICX.

This module wraps Docling so that the extractor, summariser, CLI, and API can
consume a single entry point regardless of the original file format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

try:
    from docling.document_converter import (
        DocumentConverter,
        FormatOption,
        InputFormat,
        PdfFormatOption,
        WordFormatOption,
        PowerpointFormatOption,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore[assignment]


class DocumentLoadingError(RuntimeError):
    """Raised when a document cannot be converted into text."""


@dataclass(slots=True)
class PageAnalysis:
    """Lightweight per-page stats exposed to higher layers."""

    page_number: int
    text_chars: int
    text_items: int
    picture_count: int
    ocr_score: Optional[float]


@dataclass(slots=True)
class LoadedDocument:
    """Shared representation returned by :func:`load_document`."""

    path: Path
    markdown: str
    page_analysis: List[PageAnalysis]
    docling_document: Optional[object] = None


DOC_SUFFIXES: Dict[str, InputFormat] = {
    ".pdf": InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".doc": InputFormat.DOCX,  # handled via Word pipeline
    ".pptx": InputFormat.PPTX,
    ".ppt": InputFormat.PPTX,
    ".txt": InputFormat.MD,  # treated as markdown/plain text
    ".md": InputFormat.MD,
    ".rtf": InputFormat.MD,
}

PLAIN_TEXT_SUFFIXES = {".txt", ".md", ".rtf"}


def _build_format_options() -> Dict[InputFormat, FormatOption]:
    """Return format options tuned for the formats we currently support."""
    options: Dict[InputFormat, FormatOption] = {
        InputFormat.PDF: PdfFormatOption(),
        InputFormat.DOCX: WordFormatOption(),
        InputFormat.PPTX: PowerpointFormatOption(),
    }
    return options


def _gather_page_stats(result) -> List[PageAnalysis]:
    """Derive simple per-page stats from a Docling conversion result."""
    if not getattr(result, "document", None):
        return []

    document = result.document
    page_map: Dict[int, PageAnalysis] = {}

    # Populate text statistics.
    for text_item in getattr(document, "texts", []) or []:
        prov = text_item.prov[0] if getattr(text_item, "prov", None) else None
        page_no = getattr(prov, "page_no", None)
        if page_no is None:
            continue
        stats = page_map.setdefault(
            page_no,
            PageAnalysis(
                page_number=page_no,
                text_chars=0,
                text_items=0,
                picture_count=0,
                ocr_score=None,
            ),
        )
        stats.text_chars += len(getattr(text_item, "text", "") or "")
        stats.text_items += 1

    # Populate picture counts.
    for pic in getattr(document, "pictures", []) or []:
        prov = pic.prov[0] if getattr(pic, "prov", None) else None
        page_no = getattr(prov, "page_no", None)
        if page_no is None:
            continue
        stats = page_map.setdefault(
            page_no,
            PageAnalysis(
                page_number=page_no,
                text_chars=0,
                text_items=0,
                picture_count=0,
                ocr_score=None,
            ),
        )
        stats.picture_count += 1

    # Merge OCR confidence scores if available (Docling reports per page starting at index 0).
    confidence = getattr(result, "confidence", None)
    if confidence and getattr(confidence, "pages", None):
        for index, scores in confidence.pages.items():  # type: ignore[attr-defined]
            page_number = int(index) + 1
            stats = page_map.setdefault(
                page_number,
                PageAnalysis(
                    page_number=page_number,
                    text_chars=0,
                    text_items=0,
                    picture_count=0,
                    ocr_score=None,
                ),
            )
            stats.ocr_score = getattr(scores, "ocr_score", None)

    return sorted(page_map.values(), key=lambda item: item.page_number)


def _read_plain_text(path: Path) -> str:
    """Fallback reader for simple text-based formats when Docling is unavailable."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - defensive
        raise DocumentLoadingError(f"Failed to read text from {path}: {exc}") from exc


def load_document(path: Union[str, Path]) -> LoadedDocument:
    """Convert ``path`` into a :class:`LoadedDocument` using Docling when possible."""
    doc_path = Path(path).expanduser().resolve()
    if not doc_path.exists():
        raise DocumentLoadingError(f"Document not found: {doc_path}")

    suffix = doc_path.suffix.lower()
    input_format = DOC_SUFFIXES.get(suffix)

    if input_format is None:
        raise DocumentLoadingError(
            f"Unsupported document format '{suffix}'. "
            f"Supported extensions: {', '.join(sorted(DOC_SUFFIXES))}"
        )

    # Plain-text flow without Docling if possible.
    if suffix in PLAIN_TEXT_SUFFIXES:
        text = _read_plain_text(doc_path)
        return LoadedDocument(path=doc_path, markdown=text, page_analysis=[])

    if DocumentConverter is None:
        raise DocumentLoadingError(
            "Docling is required for document conversion but is not installed."
        )

    format_options = _build_format_options()
    converter = DocumentConverter(format_options=format_options)

    try:
        result = converter.convert(doc_path)
    except Exception as exc:  # pragma: no cover - conversion errors are rare
        raise DocumentLoadingError(f"Failed to convert {doc_path}: {exc}") from exc

    markdown: str
    if getattr(result, "document", None) and hasattr(result.document, "export_to_markdown"):
        markdown = result.document.export_to_markdown()
    elif isinstance(getattr(result, "text", None), str):
        markdown = result.text  # type: ignore[assignment]
    else:
        markdown = _read_plain_text(doc_path) if input_format == InputFormat.MD else ""

    page_analysis = _gather_page_stats(result)
    return LoadedDocument(
        path=doc_path,
        markdown=markdown,
        page_analysis=page_analysis,
        docling_document=getattr(result, "document", None),
    )


__all__ = [
    "DocumentLoadingError",
    "LoadedDocument",
    "PageAnalysis",
    "load_document",
    "DOC_SUFFIXES",
]

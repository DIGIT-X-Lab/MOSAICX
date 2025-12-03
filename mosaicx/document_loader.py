"""
Unified document loading utilities for MOSAICX.

This module wraps Docling so that the extractor, summariser, CLI, and API can
consume a single entry point regardless of the original file format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from .utils.logging import get_logger

# Module-level logger
_logger = get_logger(__name__)

try:
    from docling.document_converter import (
        DocumentConverter,
        FormatOption,
        InputFormat,
        PdfFormatOption,
        WordFormatOption,
        PowerpointFormatOption,
    )
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        EasyOcrOptions,
    )
    from docling.datamodel.accelerator_options import AcceleratorOptions
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore[assignment]
    PdfPipelineOptions = None  # type: ignore[assignment]
    EasyOcrOptions = None  # type: ignore[assignment]
    AcceleratorOptions = None  # type: ignore[assignment]

try:
    import torch  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

from .constants import DEFAULT_ACCELERATOR_DEVICE


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
    conversion_result: Optional[object] = None


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


def _detect_accelerator_device() -> str:
    if DEFAULT_ACCELERATOR_DEVICE in {"cpu", "cuda", "mps"}:
        return DEFAULT_ACCELERATOR_DEVICE  # type: ignore[return-value]

    if torch is not None:
        try:
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                return "cuda"
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except Exception:
            pass

    return "cpu"


def _resolve_accelerator_options() -> Optional["AcceleratorOptions"]:
    if AcceleratorOptions is None:
        return None
    device = _detect_accelerator_device()
    return AcceleratorOptions(device=device)


def _build_format_options() -> Dict[InputFormat, FormatOption]:
    """Return format options tuned for the formats we currently support."""
    options: Dict[InputFormat, FormatOption] = {
        InputFormat.DOCX: WordFormatOption(),
        InputFormat.PPTX: PowerpointFormatOption(),
    }
    accelerator_options = _resolve_accelerator_options()
    if PdfPipelineOptions is not None and accelerator_options is not None:
        options[InputFormat.PDF] = PdfFormatOption(
            pipeline_options=PdfPipelineOptions(accelerator_options=accelerator_options)
        )
    else:
        options[InputFormat.PDF] = PdfFormatOption()
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
    _logger.debug(f"Reading plain text from {path}")
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        _logger.debug(f"Plain text read: {len(content)} chars")
        return content
    except Exception as exc:  # pragma: no cover - defensive
        _logger.error(f"Failed to read text from {path}: {exc}")
        raise DocumentLoadingError(f"Failed to read text from {path}: {exc}") from exc


def load_document(
    path: Union[str, Path],
    *,
    force_ocr: bool = False,
    languages: Optional[List[str]] = None,
    generate_images: bool = False,
    images_scale: float = 1.0,
) -> LoadedDocument:
    """Convert ``path`` into a :class:`LoadedDocument` using Docling when possible."""
    doc_path = Path(path).expanduser().resolve()
    _logger.info(f"Loading document: {doc_path}")
    _logger.debug(f"Load options: force_ocr={force_ocr}, languages={languages}, generate_images={generate_images}")
    
    if not doc_path.exists():
        _logger.error(f"Document not found: {doc_path}")
        raise DocumentLoadingError(f"Document not found: {doc_path}")

    suffix = doc_path.suffix.lower()
    input_format = DOC_SUFFIXES.get(suffix)
    _logger.debug(f"Detected format: {suffix} -> {input_format}")

    if input_format is None:
        _logger.error(f"Unsupported document format: {suffix}")
        raise DocumentLoadingError(
            f"Unsupported document format '{suffix}'. "
            f"Supported extensions: {', '.join(sorted(DOC_SUFFIXES))}"
        )

    # Plain-text flow without Docling if possible.
    if suffix in PLAIN_TEXT_SUFFIXES and not (force_ocr or generate_images):
        _logger.debug("Using plain text reader (no Docling needed)")
        text = _read_plain_text(doc_path)
        return LoadedDocument(path=doc_path, markdown=text, page_analysis=[], conversion_result=None)

    if DocumentConverter is None:
        _logger.error("Docling not installed for document conversion")
        raise DocumentLoadingError(
            "Docling is required for document conversion but is not installed."
        )

    format_options = _build_format_options()

    if input_format == InputFormat.PDF:
        pipeline_options = None
        accelerator_options = _resolve_accelerator_options()
        if force_ocr and PdfPipelineOptions is not None and EasyOcrOptions is not None:
            lang_list = languages or ["en", "de"]
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,
                generate_page_images=generate_images,
                images_scale=images_scale,
                ocr_options=EasyOcrOptions(
                    lang=lang_list,
                    force_full_page_ocr=True,
                    bitmap_area_threshold=0.0,
                    use_gpu=None,
                ),
                accelerator_options=accelerator_options,
            )
        elif generate_images and PdfPipelineOptions is not None:
            pipeline_options = PdfPipelineOptions(
                generate_page_images=True,
                images_scale=images_scale,
                accelerator_options=accelerator_options,
            )

        if pipeline_options is not None:
            format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pipeline_options)
    converter = DocumentConverter(format_options=format_options)
    _logger.debug("Docling DocumentConverter initialized")

    try:
        _logger.info(f"Converting document with Docling: {doc_path.name}")
        result = converter.convert(doc_path)
        _logger.debug("Docling conversion completed successfully")
    except Exception as exc:  # pragma: no cover - conversion errors are rare
        _logger.error(f"Docling conversion failed for {doc_path}: {exc}")
        raise DocumentLoadingError(f"Failed to convert {doc_path}: {exc}") from exc

    markdown: str
    if getattr(result, "document", None) and hasattr(result.document, "export_to_markdown"):
        markdown = result.document.export_to_markdown()
        _logger.debug(f"Exported markdown from Docling document: {len(markdown)} chars")
    elif isinstance(getattr(result, "text", None), str):
        markdown = result.text  # type: ignore[assignment]
        _logger.debug(f"Used raw text from result: {len(markdown)} chars")
    else:
        markdown = _read_plain_text(doc_path) if input_format == InputFormat.MD else ""
        _logger.debug(f"Fallback to plain text: {len(markdown)} chars")

    page_analysis = _gather_page_stats(result)
    _logger.info(f"Document loaded: {len(markdown)} chars, {len(page_analysis)} pages analyzed")
    
    return LoadedDocument(
        path=doc_path,
        markdown=markdown,
        page_analysis=page_analysis,
        docling_document=getattr(result, "document", None),
        conversion_result=result,
    )


__all__ = [
    "DocumentLoadingError",
    "LoadedDocument",
    "PageAnalysis",
    "load_document",
    "DOC_SUFFIXES",
]

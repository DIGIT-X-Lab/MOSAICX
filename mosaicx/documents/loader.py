"""
Unified document loading — converts PDF, DOCX, PPTX, Markdown, and plain text
into a LoadedDocument with extracted text.
Uses Docling for structured formats; falls back to plain read for .txt/.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

_DOCLING_FORMATS = {".pdf", ".docx", ".pptx"}
_TEXT_FORMATS = {".txt", ".md", ".markdown"}
_ALL_SUPPORTED = _DOCLING_FORMATS | _TEXT_FORMATS


@dataclass
class LoadedDocument:
    """A document converted to plain text."""
    text: str
    source_path: Path
    format: str
    page_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


def load_document(path: Path) -> LoadedDocument:
    """Load a document from disk and convert to text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in _ALL_SUPPORTED:
        raise ValueError(f"Unsupported format '{suffix}'. Supported: {sorted(_ALL_SUPPORTED)}")
    if suffix in _TEXT_FORMATS:
        return _load_text(path, suffix.lstrip("."))
    return _load_with_docling(path, suffix.lstrip("."))


def _load_text(path: Path, fmt: str) -> LoadedDocument:
    text = path.read_text(encoding="utf-8")
    return LoadedDocument(text=text, source_path=path, format=fmt)


def _load_with_docling(path: Path, fmt: str) -> LoadedDocument:
    if DocumentConverter is None:
        raise RuntimeError("Docling is required for PDF/DOCX/PPTX. Install with: pip install docling")
    converter = DocumentConverter()
    result = converter.convert(str(path))
    text = result.document.export_to_markdown()
    page_count = getattr(result.document, "page_count", None)
    return LoadedDocument(text=text, source_path=path, format=fmt, page_count=page_count)

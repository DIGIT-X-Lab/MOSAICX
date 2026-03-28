# mosaicx/documents/models.py
"""Document data models for the loading pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


@dataclass
class TextBlock:
    """A contiguous text region with its location on the source page."""

    text: str
    start: int  # char offset in full document text
    end: int  # char offset in full document text
    page: int  # 1-indexed page number
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in page points


@dataclass
class PageResult:
    """OCR result for a single page."""

    page_number: int
    text: str
    engine: str
    confidence: float
    layout_html: Optional[str] = None
    text_blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class LoadedDocument:
    """A document converted to plain text with OCR metadata."""

    text: str
    source_path: Path
    format: str
    page_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    ocr_engine_used: Optional[str] = None
    ocr_confidence: Optional[float] = None
    quality_warning: bool = False
    pages: list[PageResult] = field(default_factory=list)
    text_blocks: list[TextBlock] = field(default_factory=list)
    page_dimensions: list[tuple[float, float]] = field(default_factory=list)
    _ocr_preprocessed_img: Any = field(default=None, repr=False)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0

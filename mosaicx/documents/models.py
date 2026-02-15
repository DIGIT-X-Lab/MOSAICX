# mosaicx/documents/models.py
"""Document data models for the loading pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


@dataclass
class PageResult:
    """OCR result for a single page."""

    page_number: int
    text: str
    engine: str
    confidence: float
    layout_html: Optional[str] = None


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

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0

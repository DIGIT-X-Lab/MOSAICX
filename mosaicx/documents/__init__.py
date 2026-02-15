"""Document loading — dual-engine OCR (Surya + Chandra) + plain text."""
from .models import LoadedDocument, PageResult, DocumentLoadError
from .loader import load_document

__all__ = ["LoadedDocument", "PageResult", "DocumentLoadError", "load_document"]

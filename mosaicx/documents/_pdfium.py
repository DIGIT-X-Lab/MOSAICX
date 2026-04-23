"""Shared lock for pypdfium2 / PDFium calls.

PDFium (the underlying C library behind pypdfium2) is not thread-safe.
Any code path that opens a ``PdfDocument`` or walks its pages must
acquire this lock to serialize access across threads. Locks are held
only for the duration of PDFium calls (open, iterate pages, render,
get textpage); release before doing network OCR work so unrelated
threads can rasterize their own PDFs concurrently.
"""

from __future__ import annotations

import threading

PDFIUM_LOCK: threading.Lock = threading.Lock()

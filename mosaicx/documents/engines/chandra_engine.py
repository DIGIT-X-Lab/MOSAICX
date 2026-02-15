# mosaicx/documents/engines/chandra_engine.py
"""Chandra OCR engine wrapper.

Chandra is a VLM-based OCR (fine-tuned Qwen3-VL 9B) that handles
handwriting, complex forms, tables, and mixed layouts. Supports
vLLM (CUDA) and HuggingFace transformers (CUDA/MPS) backends.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from PIL import Image

from ..models import PageResult

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def _detect_backend() -> str:
    """Auto-detect best backend: vLLM on CUDA, HF on MPS/CPU."""
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "vllm"
    return "hf"


@lru_cache(maxsize=1)
def _get_chandra_manager(backend: str | None = None):
    """Lazily load and cache the Chandra inference manager."""
    from chandra.model import InferenceManager

    method = backend or _detect_backend()
    return InferenceManager(method=method)


class ChandraEngine:
    """OCR engine backed by Chandra VLM."""

    def __init__(self, backend: str | None = None, server_url: str | None = None):
        self._backend = backend
        self._server_url = server_url

    def ocr_pages(
        self,
        images: list[Image.Image],
        langs: list[str] | None = None,
    ) -> list[PageResult]:
        if not images:
            return []

        manager = _get_chandra_manager(self._backend)
        try:
            results = manager.generate(images)
        except Exception:
            logger.exception("Chandra OCR failed")
            return [
                PageResult(page_number=i + 1, text="", engine="chandra", confidence=0.0)
                for i in range(len(images))
            ]

        page_results = []
        for i, result in enumerate(results):
            text = getattr(result, "markdown", "") or ""
            html = getattr(result, "html", None)
            page_results.append(
                PageResult(
                    page_number=i + 1,
                    text=text,
                    engine="chandra",
                    confidence=0.9,  # Chandra doesn't expose per-page confidence
                    layout_html=html,
                )
            )
        return page_results

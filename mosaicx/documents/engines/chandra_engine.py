# mosaicx/documents/engines/chandra_engine.py
"""Chandra OCR engine wrapper.

Chandra is a VLM-based OCR (fine-tuned Qwen3-VL 9B) that handles
handwriting, complex forms, tables, and mixed layouts. Supports
vLLM (CUDA) and HuggingFace transformers (CUDA/MPS) backends.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from PIL import Image

from ..models import PageResult

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def _patch_chandra_http_timeout() -> None:
    """Add a client-side HTTP timeout to chandra's bundled OpenAI client.

    Chandra 0.1.x constructs its ``OpenAI(...)`` inside ``chandra.model.vllm``
    without a ``timeout=`` argument. The OpenAI SDK then falls back to its
    600s default, so if a pooled socket is stale (vLLM server-side keep-alive
    has closed it, client hasn't noticed) the request blocks for up to ten
    minutes per retry — and chandra retries up to ``MAX_VLLM_RETRIES=6``
    times. With a single-threaded OCR producer this stalls the entire batch.

    We subclass ``openai.OpenAI`` to inject a per-request timeout and cut
    ``max_retries`` to 1, then replace the symbol in ``chandra.model.vllm``.
    Timeout is configurable via ``MOSAICX_CHANDRA_HTTP_TIMEOUT`` (seconds,
    default 60).

    Safe to call repeatedly: wrapping is idempotent.
    """
    try:
        import chandra.model.vllm as _cv
        import openai
    except ImportError:
        return

    if getattr(_cv.OpenAI, "_mosaicx_patched", False):
        return

    try:
        timeout_seconds = float(os.environ.get("MOSAICX_CHANDRA_HTTP_TIMEOUT", "60"))
    except (TypeError, ValueError):
        timeout_seconds = 60.0

    orig_openai = _cv.OpenAI

    class _OpenAIWithTimeout(orig_openai):  # type: ignore[misc,valid-type]
        _mosaicx_patched = True

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("timeout", timeout_seconds)
            kwargs.setdefault("max_retries", 1)
            super().__init__(*args, **kwargs)

    _cv.OpenAI = _OpenAIWithTimeout
    logger.info(
        "Patched chandra.model.vllm.OpenAI with timeout=%ss, max_retries=1",
        timeout_seconds,
    )


def _detect_backend() -> str:
    """Auto-detect best backend: vLLM on CUDA, HF on MPS/CPU."""
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "vllm"
    return "hf"


def _patch_chandra_hf_device() -> None:
    """Patch chandra's HF backend to use the correct device (MPS/CPU).

    Chandra 0.1.8 hardcodes ``inputs.to("cuda")`` in ``generate_hf``.
    We replace the function in both the module and the caller namespace.
    """
    try:
        import chandra.model as _cm
        import chandra.model.hf as _hf

        from chandra.model.hf import (
            GenerationResult,
            process_batch_element,
            process_vision_info,
            settings,
        )

        def _generate_hf(batch, model, max_output_tokens=None, **kwargs):
            if max_output_tokens is None:
                max_output_tokens = settings.MAX_OUTPUT_TOKENS

            device = next(model.parameters()).device

            messages = [process_batch_element(item, model.processor) for item in batch]
            text = model.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = model.processor(
                text=text,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                padding_side="left",
            )
            inputs = inputs.to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=max_output_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = model.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return [
                GenerationResult(raw=out, token_count=len(ids), error=False)
                for out, ids in zip(output_text, generated_ids_trimmed)
            ]

        # Patch in both places so InferenceManager.generate picks it up
        _hf.generate_hf = _generate_hf
        _cm.generate_hf = _generate_hf
    except (ImportError, Exception):
        pass


@lru_cache(maxsize=1)
def _get_chandra_manager(backend: str | None = None, server_url: str | None = None):
    """Lazily load and cache the Chandra inference manager."""
    from chandra.model import InferenceManager

    # If a server URL is provided, always use vLLM backend (remote server)
    if server_url:
        from chandra.model import settings as chandra_settings
        chandra_settings.VLLM_API_BASE = server_url
        # Wrap the bundled OpenAI client with a sane timeout before the
        # first InferenceManager is created; otherwise stuck requests
        # block the OCR producer for up to 600s each.
        _patch_chandra_http_timeout()
        return InferenceManager(method="vllm")

    method = backend or _detect_backend()
    if method == "hf":
        _patch_chandra_hf_device()
    else:
        _patch_chandra_http_timeout()
    return InferenceManager(method=method)


class ChandraEngine:
    """OCR engine backed by Chandra VLM.

    Supports three modes:
    - **vLLM server** (recommended): Set ``server_url`` to a running Chandra
      vLLM server (e.g. ``http://gpu-server:8000/v1``). Lightweight, no
      local model loading.
    - **vLLM local**: Auto-detected when CUDA is available.
    - **HuggingFace**: Loads the 9B model in-process. Needs ~20GB RAM.
    """

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

        if not self._server_url:
            raise RuntimeError(
                "Chandra requires a vLLM server. Set MOSAICX_CHANDRA_SERVER_URL "
                "in your .env file (e.g. http://localhost:8000/v1).\n\n"
                "To start a Chandra server:\n"
                "  pip install chandra-ocr\n"
                "  chandra_vllm\n\n"
                "Or use PaddleOCR instead: MOSAICX_OCR_ENGINE=paddleocr"
            )

        manager = _get_chandra_manager(self._backend, self._server_url)
        try:
            from chandra.model import BatchInputItem

            batch = [BatchInputItem(image=img, prompt_type="ocr") for img in images]
            kwargs = {}
            if self._server_url:
                kwargs["vllm_api_base"] = self._server_url
            results = manager.generate(batch, **kwargs)
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
                    confidence=0.9,
                    layout_html=html,
                )
            )
        return page_results

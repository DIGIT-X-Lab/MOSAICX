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
def _get_chandra_manager(backend: str | None = None):
    """Lazily load and cache the Chandra inference manager."""
    from chandra.model import InferenceManager

    _patch_chandra_hf_device()
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
            from chandra.model import BatchInputItem

            batch = [BatchInputItem(image=img, prompt_type="ocr") for img in images]
            results = manager.generate(batch)
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

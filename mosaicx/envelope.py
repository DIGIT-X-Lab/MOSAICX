# mosaicx/envelope.py
"""
Unified _mosaicx metadata envelope builder.

Every pipeline output is wrapped with a ``_mosaicx`` key containing the dict
returned by :func:`build_envelope`.  This replaces the previous ad-hoc
``_metrics`` and ``_document`` keys with a single, always-present block.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _get_version() -> str:
    """Return the installed MOSAICX package version, with fallback."""
    try:
        from importlib.metadata import version

        return version("mosaicx")
    except Exception:
        return "2.0.0a1"


def _get_model_info() -> tuple[str, float]:
    """Read model name and temperature from MosaicxConfig.

    Returns a safe fallback when config is unavailable (e.g. in tests).
    """
    try:
        from mosaicx.config import get_config

        cfg = get_config()
        return cfg.lm, cfg.lm_temperature
    except Exception:
        return "unknown", 0.0


def build_envelope(
    *,
    pipeline: str,
    template: str | None = None,
    template_version: str | None = None,
    duration_s: float | None = None,
    tokens: dict[str, int] | None = None,
    provenance: bool = False,
    verification: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable metadata envelope.

    Parameters
    ----------
    pipeline:
        Name of the pipeline that produced the output (required).
    template:
        Template name used for extraction, or ``None``.
    template_version:
        Version string of the template, or ``None``.
    duration_s:
        Wall-clock processing time in seconds, or ``None``.
    tokens:
        Dict with ``input`` and ``output`` token counts.
        Defaults to ``{"input": 0, "output": 0}``.
    provenance:
        Whether provenance tracking was enabled.
    verification:
        Summary dict from the verification step, or ``None``.
    document:
        Document metadata dict (filename, pages, etc.), or ``None``.

    Returns
    -------
    dict
        A flat, JSON-serializable dict suitable for the ``_mosaicx`` key.
    """
    model, model_temperature = _get_model_info()

    return {
        "version": _get_version(),
        "pipeline": pipeline,
        "template": template,
        "template_version": template_version,
        "model": model,
        "model_temperature": model_temperature,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "tokens": tokens if tokens is not None else {"input": 0, "output": 0},
        "provenance": provenance,
        "verification": verification,
        "document": document,
    }

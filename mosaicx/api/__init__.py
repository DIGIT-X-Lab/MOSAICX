"""
MOSAICX Programmatic API — Harnessing Clinical AI Pipelines in Python

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
DIGIT-X Lab, LMU Radiology | Lalith Kumar Shiyam Sundar, PhD
================================================================================

Overview
--------
This subpackage exposes the same capabilities as the MOSAICX CLI—schema
generation, document extraction, longitudinal summarisation—while remaining free of
terminal side effects.  Use it inside notebooks, production services, or test
harnesses to drive the MOSAICX pipeline through a clean, composable API.

Endpoint Configuration
----------------------
Every helper accepts optional ``base_url`` and ``api_key`` parameters.  When
omitted, the implementation resolves them in order of precedence:

1. Explicit keyword argument supplied by the caller
2. ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` environment variables
3. Local Ollama-compatible defaults (``http://localhost:11434/v1`` + ``"ollama"``)

These defaults mirror the CLI behaviour, ensuring consistent results across
automation and interactive sessions.
"""

from .schema import GeneratedSchema, generate_schema
from .extraction import ExtractionResult, extract_document, extract_pdf
from .summary import summarize_reports
from ..standardizer import StandardizeResult, save_standardize_artifacts, standardize_document

__all__ = [
    "GeneratedSchema",
    "generate_schema",
    "ExtractionResult",
    "extract_document",
    "extract_pdf",
    "summarize_reports",
    "StandardizeResult",
    "standardize_document",
    "save_standardize_artifacts",
]

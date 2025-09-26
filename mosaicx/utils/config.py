"""
MOSAICX Configuration Utilities - OpenAI-Compatible Endpoints

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Structure first. Insight follows.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab
Department: Department of Radiology
University: LMU University Hospital | LMU Munich

Overview:
---------
Resolve endpoint metadata for the LLM integrations used across schema
generation, extraction, and summarisation.  Functions here encapsulate common
fallback rules so both the CLI and API consume identical defaults.

Capabilities:
------------
- Merge explicit CLI arguments, environment variables, and project defaults.
- Detect local Ollama hosts and surface their legacy ``/api/generate`` URL for
  compatibility with streaming endpoints.
- Normalise URLs to avoid trailing slashes while preserving user intent.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from urllib.parse import urlparse, urlunparse

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"


def resolve_openai_config(
    base_url: Optional[str],
    api_key: Optional[str],
    *,
    default_base_url: str = DEFAULT_BASE_URL,
    default_api_key: str = DEFAULT_API_KEY,
) -> Tuple[str, str]:
    """Resolve endpoint configuration using overrides, env vars, and defaults."""

    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or default_base_url
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or default_api_key
    return resolved_base_url, resolved_api_key


def derive_ollama_generate_url(base_url: str) -> Optional[str]:
    """Return /api/generate endpoint if the base URL indicates a local Ollama host."""

    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    if not host:
        return None
    if not any(token in host for token in ("localhost", "127.0.0.1", "ollama")):
        return None
    root = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
    if not root:
        return None
    return root.rstrip("/") + "/api/generate"


__all__ = ["resolve_openai_config", "derive_ollama_generate_url"]

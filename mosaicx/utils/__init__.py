"""
MOSAICX Utilities Package - Cross-Cutting Helpers

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
Central repository for helper functions reused by the CLI, API, and background
pipelines.  Utility modules provide filesystem resolution, endpoint
configuration, and integration glue without importing heavier dependencies at
package load time.

Focus Areas:
-----------
- Schema path discovery for generated Pydantic models.
- Endpoint discovery for OpenAI-compatible services and local Ollama setups.
- Forwarding of lightweight convenience functions through ``__all__`` to keep
  intra-package imports clean and intention revealing.
"""

from .pathing import resolve_schema_reference
from .config import resolve_openai_config, derive_ollama_generate_url
from .logging import (
    setup_logging,
    get_logger,
    get_current_log_file,
    get_session_id,
    log_extraction_start,
    log_extraction_method_attempt,
    log_extraction_method_success,
    log_extraction_method_failure,
    log_prompt,
    log_llm_response,
    log_extraction_complete,
    log_schema_info,
    log_text_content,
)

__all__ = [
    "resolve_schema_reference",
    "resolve_openai_config",
    "derive_ollama_generate_url",
    # Logging utilities
    "setup_logging",
    "get_logger",
    "get_current_log_file",
    "get_session_id",
    "log_extraction_start",
    "log_extraction_method_attempt",
    "log_extraction_method_success",
    "log_extraction_method_failure",
    "log_prompt",
    "log_llm_response",
    "log_extraction_complete",
    "log_schema_info",
    "log_text_content",
]

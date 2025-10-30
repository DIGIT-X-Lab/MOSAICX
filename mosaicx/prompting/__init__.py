"""
Prompt management utilities for MOSAICX extraction strategies.

The prompt subsystem is responsible for synthesising schema-aware base prompts,
persisting optimised variants, and serving the correct instructions to the
extraction pipeline at runtime.
"""

from __future__ import annotations

from .manager import (
    PromptArtifact,
    PromptPreference,
    PromptVariant,
    resolve_prompt_for_schema,
    synthesise_base_prompt,
    build_example_template,
)

__all__ = [
    "PromptArtifact",
    "PromptPreference",
    "PromptVariant",
    "resolve_prompt_for_schema",
    "synthesise_base_prompt",
    "build_example_template",
]

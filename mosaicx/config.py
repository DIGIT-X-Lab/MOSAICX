# mosaicx/config.py
"""
MOSAICX Configuration — Single source of truth via Pydantic Settings.

Resolution order: CLI flags > env vars (MOSAICX_*) > config file > defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MosaicxConfig(BaseSettings):
    """Central configuration for MOSAICX v2."""

    model_config = SettingsConfigDict(
        env_prefix="MOSAICX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    lm: str = "ollama_chat/llama3.1:70b"
    lm_cheap: str = "ollama_chat/llama3.2:3b"
    api_key: str = "ollama"

    # --- Processing ---
    default_template: str = "auto"
    completeness_threshold: float = 0.7
    batch_workers: int = 4
    checkpoint_every: int = 50

    # --- Paths ---
    home_dir: Path = Field(default_factory=lambda: Path.home() / ".mosaicx")

    @property
    def schema_dir(self) -> Path:
        return self.home_dir / "schemas"

    @property
    def optimized_dir(self) -> Path:
        return self.home_dir / "optimized"

    @property
    def checkpoint_dir(self) -> Path:
        return self.home_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.home_dir / "logs"

    # --- De-identification ---
    deidentify_mode: Literal["remove", "pseudonymize", "dateshift"] = "remove"

    # --- Export ---
    default_export_formats: list[str] = Field(
        default_factory=lambda: ["parquet", "jsonl"]
    )

    # --- Document loading ---
    force_ocr: bool = False
    ocr_langs: list[str] = Field(default_factory=lambda: ["en", "de"])
    vlm_model: str = "gemma3:27b"


@lru_cache(maxsize=1)
def get_config() -> MosaicxConfig:
    """Return the global config singleton."""
    return MosaicxConfig()

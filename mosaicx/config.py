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
    lm: str = "openai/gpt-oss:120b"
    lm_cheap: str = "openai/gpt-oss:20b"
    api_key: str = "ollama"
    api_base: str = "http://localhost:11434/v1"
    lm_temperature: float = 0.0

    # --- Processing ---
    default_template: str = "auto"
    completeness_threshold: float = 0.7
    batch_workers: int = 1
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
    ocr_engine: Literal["both", "surya", "chandra"] = "both"
    chandra_backend: Literal["vllm", "hf", "auto"] = "auto"
    chandra_server_url: str = ""
    quality_threshold: float = 0.6
    ocr_page_timeout: int = 60
    force_ocr: bool = False
    ocr_langs: list[str] = Field(default_factory=lambda: ["en", "de"])


@lru_cache(maxsize=1)
def get_config() -> MosaicxConfig:
    """Return the global config singleton."""
    return MosaicxConfig()

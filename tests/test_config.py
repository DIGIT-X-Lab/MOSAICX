# tests/test_config.py
"""Tests for MosaicxConfig — Pydantic Settings single source of truth."""

import os
import pytest
from pathlib import Path


class TestMosaicxConfig:
    """Test MosaicxConfig defaults and overrides."""

    def test_default_values(self):
        """Config should have sensible defaults without any env vars."""
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig()
        assert cfg.lm == "openai/gpt-oss:120b"
        assert cfg.lm_cheap == "openai/gpt-oss:20b"
        assert cfg.completeness_threshold == 0.7
        assert cfg.batch_workers == 4
        assert cfg.checkpoint_every == 50
        assert cfg.default_template == "auto"
        assert cfg.deidentify_mode == "remove"
        assert cfg.default_export_formats == ["parquet", "jsonl"]
        assert cfg.force_ocr is False
        assert cfg.ocr_langs == ["en", "de"]

    def test_env_override(self, monkeypatch):
        """Environment variables with MOSAICX_ prefix override defaults."""
        from mosaicx.config import MosaicxConfig

        monkeypatch.setenv("MOSAICX_LM", "openai/gpt-4o")
        monkeypatch.setenv("MOSAICX_BATCH_WORKERS", "16")
        cfg = MosaicxConfig()
        assert cfg.lm == "openai/gpt-4o"
        assert cfg.batch_workers == 16

    def test_home_dir_default(self):
        """home_dir defaults to ~/.mosaicx."""
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig()
        assert cfg.home_dir == Path.home() / ".mosaicx"

    def test_derived_paths(self):
        """schema_dir, optimized_dir, etc. derive from home_dir."""
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig()
        assert cfg.schema_dir == cfg.home_dir / "schemas"
        assert cfg.optimized_dir == cfg.home_dir / "optimized"
        assert cfg.checkpoint_dir == cfg.home_dir / "checkpoints"
        assert cfg.log_dir == cfg.home_dir / "logs"

    def test_ocr_config_defaults(self):
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig()
        assert cfg.ocr_engine == "both"
        assert cfg.chandra_backend == "auto"
        assert cfg.quality_threshold == 0.6
        assert cfg.ocr_page_timeout == 60
        assert cfg.force_ocr is False
        assert cfg.ocr_langs == ["en", "de"]

    def test_ocr_engine_env_override(self, monkeypatch):
        from mosaicx.config import MosaicxConfig

        monkeypatch.setenv("MOSAICX_OCR_ENGINE", "surya")
        cfg = MosaicxConfig()
        assert cfg.ocr_engine == "surya"

    def test_get_config_singleton(self):
        """get_config() returns the same instance."""
        from mosaicx.config import get_config

        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

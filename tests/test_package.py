# tests/test_package.py
"""Tests for top-level package API."""

import pytest


class TestPackageImports:
    """Verify the public API surface."""

    def test_version(self):
        import mosaicx
        assert hasattr(mosaicx, "__version__")
        assert "2.0" in mosaicx.__version__

    def test_config_importable(self):
        from mosaicx.config import MosaicxConfig, get_config
        assert MosaicxConfig is not None
        assert callable(get_config)

    def test_cli_importable(self):
        from mosaicx.cli import cli
        assert callable(cli)

    def test_documents_importable(self):
        from mosaicx.documents import LoadedDocument, load_document
        assert LoadedDocument is not None
        assert callable(load_document)

# tests/test_documents.py
"""Tests for the document loading module."""

import pytest
from pathlib import Path


class TestLoadedDocument:
    """Test the LoadedDocument dataclass."""

    def test_construction(self):
        from mosaicx.documents.loader import LoadedDocument

        doc = LoadedDocument(
            text="Hello world",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            page_count=3,
        )
        assert doc.text == "Hello world"
        assert doc.format == "pdf"
        assert doc.page_count == 3

    def test_char_count(self):
        from mosaicx.documents.loader import LoadedDocument

        doc = LoadedDocument(text="abc", source_path=Path("/tmp/x.pdf"), format="pdf")
        assert doc.char_count == 3

    def test_is_empty(self):
        from mosaicx.documents.loader import LoadedDocument

        empty = LoadedDocument(text="", source_path=Path("/tmp/x.pdf"), format="pdf")
        nonempty = LoadedDocument(text="content", source_path=Path("/tmp/x.pdf"), format="pdf")
        assert empty.is_empty is True
        assert nonempty.is_empty is False


class TestLoadDocument:
    """Test load_document function."""

    def test_unsupported_format_raises(self, tmp_path):
        from mosaicx.documents.loader import load_document

        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(bad_file)

    def test_missing_file_raises(self):
        from mosaicx.documents.loader import load_document

        with pytest.raises(FileNotFoundError):
            load_document(Path("/nonexistent/file.pdf"))

    def test_plain_text_loading(self, tmp_path):
        from mosaicx.documents.loader import load_document

        txt_file = tmp_path / "report.txt"
        txt_file.write_text("Patient presents with cough.")
        doc = load_document(txt_file)
        assert "Patient presents with cough" in doc.text
        assert doc.format == "txt"

    def test_markdown_loading(self, tmp_path):
        from mosaicx.documents.loader import load_document

        md_file = tmp_path / "report.md"
        md_file.write_text("# Findings\n\nNormal chest.")
        doc = load_document(md_file)
        assert "Normal chest" in doc.text
        assert doc.format == "md"

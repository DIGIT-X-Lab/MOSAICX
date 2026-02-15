# tests/test_export_report.py
"""Tests for narrative report export."""
import pytest

class TestMarkdownExport:
    def test_export_markdown(self, tmp_path):
        from mosaicx.export.report import export_markdown
        result = {
            "source_file": "report.pdf",
            "exam_type": "chest_ct",
            "indication": "Cough",
            "findings": [{"anatomy": "RUL", "observation": "nodule", "description": "5mm nodule"}],
            "impression": "Pulmonary nodule.",
            "completeness": 0.85,
        }
        path = tmp_path / "report.md"
        export_markdown(result, path)
        assert path.exists()
        content = path.read_text()
        assert "chest_ct" in content or "Chest" in content
        assert "nodule" in content

    def test_export_markdown_empty_findings(self, tmp_path):
        from mosaicx.export.report import export_markdown
        result = {"source_file": "r.pdf", "exam_type": "generic",
                  "findings": [], "impression": "Normal.", "completeness": 1.0}
        path = tmp_path / "report.md"
        export_markdown(result, path)
        assert path.exists()

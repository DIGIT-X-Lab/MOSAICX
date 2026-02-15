# tests/test_export_tabular.py
"""Tests for tabular export."""
import pytest
import json
from pathlib import Path

SAMPLE_RESULTS = [
    {
        "source_file": "report1.pdf",
        "exam_type": "chest_ct",
        "indication": "Cough",
        "findings": [
            {"anatomy": "RUL", "observation": "nodule", "size_mm": 5.0},
            {"anatomy": "RLL", "observation": "atelectasis"},
        ],
        "impression": "Pulmonary nodule.",
        "completeness": 0.85,
    },
    {
        "source_file": "report2.pdf",
        "exam_type": "chest_ct",
        "indication": "Follow-up",
        "findings": [{"anatomy": "RUL", "observation": "nodule", "size_mm": 5.0}],
        "impression": "Stable nodule.",
        "completeness": 0.90,
    },
]

class TestCSVExport:
    def test_one_row_strategy(self, tmp_path):
        from mosaicx.export.tabular import export_csv
        path = tmp_path / "output.csv"
        export_csv(SAMPLE_RESULTS, path, strategy="one_row")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_findings_rows_strategy(self, tmp_path):
        from mosaicx.export.tabular import export_csv
        path = tmp_path / "output.csv"
        export_csv(SAMPLE_RESULTS, path, strategy="findings_rows")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 finding rows

class TestJSONLExport:
    def test_export_jsonl(self, tmp_path):
        from mosaicx.export.tabular import export_jsonl
        path = tmp_path / "output.jsonl"
        export_jsonl(SAMPLE_RESULTS, path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "source_file" in obj

class TestParquetExport:
    def test_export_parquet(self, tmp_path):
        from mosaicx.export.tabular import export_parquet
        path = tmp_path / "output.parquet"
        export_parquet(SAMPLE_RESULTS, path, strategy="one_row")
        assert path.exists()
        assert path.stat().st_size > 0

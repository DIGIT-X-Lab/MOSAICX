# tests/test_query_loaders.py
"""Tests for query source loaders."""

from __future__ import annotations

import json
from pathlib import Path


class TestJSONLoader:
    def test_load_json_file(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "value"}))
        meta, data = load_source(f)
        assert meta.source_type == "json"
        assert data["key"] == "value"

    def test_load_json_list(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "items.json"
        f.write_text(json.dumps([1, 2, 3]))
        meta, data = load_source(f)
        assert meta.source_type == "json"
        assert data == [1, 2, 3]

    def test_json_meta_fields(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"a": 1}))
        meta, _ = load_source(f)
        assert meta.name == "data.json"
        assert meta.format == "json"
        assert meta.size > 0
        assert meta.preview is not None


class TestCSVLoader:
    def test_load_csv_file(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        meta, data = load_source(f)
        assert meta.source_type == "dataframe"
        assert len(data) == 2  # 2 rows

    def test_csv_preview_contains_columns(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\n")
        meta, _ = load_source(f)
        assert "name" in meta.preview
        assert "age" in meta.preview


class TestParquetLoader:
    def test_load_parquet_file(self, tmp_path: Path):
        import pandas as pd

        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.parquet"
        pd.DataFrame({"col": [1, 2, 3]}).to_parquet(f)
        meta, data = load_source(f)
        assert meta.source_type == "dataframe"
        assert len(data) == 3

    def test_parquet_meta(self, tmp_path: Path):
        import pandas as pd

        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.parquet"
        pd.DataFrame({"x": [10], "y": [20]}).to_parquet(f)
        meta, _ = load_source(f)
        assert meta.format == "parquet"
        assert "2 cols" in meta.preview


class TestTextLoader:
    def test_load_txt_file(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "report.txt"
        f.write_text("Patient presents with cough.")
        meta, data = load_source(f)
        assert meta.source_type == "document"
        assert "cough" in data

    def test_load_md_file(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "notes.md"
        f.write_text("# Heading\nSome content here.")
        meta, data = load_source(f)
        assert meta.source_type == "document"
        assert meta.format == "md"
        assert "Heading" in data

    def test_unknown_extension_falls_back_to_text(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.log"
        f.write_text("log line 1\nlog line 2\n")
        meta, data = load_source(f)
        assert meta.source_type == "document"
        assert "log line 1" in data


class TestSourceMeta:
    def test_meta_has_required_fields(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        meta, _ = load_source(f)
        assert meta.name == "data.csv"
        assert meta.format == "csv"
        assert meta.size > 0
        assert meta.preview is not None

    def test_meta_preview_truncated(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "big.txt"
        f.write_text("x" * 500)
        meta, _ = load_source(f)
        assert len(meta.preview) <= 200


class TestLoadSourcePath:
    def test_accepts_string_path(self, tmp_path: Path):
        from mosaicx.query.loaders import load_source

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"hello": "world"}))
        meta, data = load_source(str(f))
        assert meta.source_type == "json"
        assert data["hello"] == "world"

# tests/test_query_tools.py
from __future__ import annotations

import json
from pathlib import Path


class TestSearchDocuments:
    def test_keyword_search(self):
        from mosaicx.query.tools import search_documents

        docs = {"report1.txt": "Patient has 5mm nodule in RUL.", "report2.txt": "Normal chest X-ray."}
        results = search_documents("nodule", documents=docs, top_k=5)
        assert len(results) >= 1
        assert "report1.txt" in results[0]["source"]

    def test_no_results(self):
        from mosaicx.query.tools import search_documents

        docs = {"report1.txt": "Normal exam."}
        results = search_documents("nodule", documents=docs)
        assert len(results) == 0

    def test_top_k_limits(self):
        from mosaicx.query.tools import search_documents

        docs = {f"doc{i}.txt": f"Finding number {i}" for i in range(10)}
        results = search_documents("finding", documents=docs, top_k=3)
        assert len(results) == 3


class TestGetDocument:
    def test_get_by_name(self):
        from mosaicx.query.tools import get_document

        docs = {"report1.txt": "Patient has cough."}
        text = get_document("report1.txt", documents=docs)
        assert "cough" in text

    def test_get_missing_raises(self):
        from mosaicx.query.tools import get_document
        import pytest

        with pytest.raises(KeyError):
            get_document("nonexistent.txt", documents={})


class TestSaveArtifact:
    def test_save_csv(self, tmp_path: Path):
        from mosaicx.query.tools import save_artifact

        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        path = save_artifact(data, tmp_path / "output.csv", format="csv")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Alice" in content

    def test_save_json(self, tmp_path: Path):
        from mosaicx.query.tools import save_artifact

        data = {"key": "value", "items": [1, 2, 3]}
        path = save_artifact(data, tmp_path / "output.json", format="json")
        assert Path(path).exists()
        loaded = json.loads(Path(path).read_text())
        assert loaded["key"] == "value"

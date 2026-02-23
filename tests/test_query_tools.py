# tests/test_query_tools.py
from __future__ import annotations

import json
from pathlib import Path

import pytest


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

    def test_ignores_short_substring_false_positives(self):
        from mosaicx.query.tools import search_documents

        docs = {"report.txt": "Technique: contrast-enhanced CT of chest."}
        results = search_documents("can", documents=docs, top_k=5)
        assert len(results) == 0

    def test_plural_query_matches_singular_text(self):
        from mosaicx.query.tools import search_documents

        docs = {"report.txt": "The radiology note reports no pleural effusion."}
        results = search_documents("summarize reports", documents=docs, top_k=5)
        assert len(results) >= 1
        assert results[0]["source"] == "report.txt"

    def test_prefers_line_with_more_query_terms(self):
        from mosaicx.query.tools import search_documents

        docs = {
            "report.txt": (
                "Lungs: No suspicious pulmonary nodules.\n"
                "Lymph nodes: Right external iliac node increased in size (now 16 mm short-axis)."
            )
        }
        results = search_documents("lesion size change lymph node", documents=docs, top_k=3)
        assert len(results) >= 1
        assert "16 mm" in results[0]["snippet"]

    def test_cancer_query_matches_carcinoma_synonym(self):
        from mosaicx.query.tools import search_documents

        docs = {"report.txt": "Indication: Known prostate carcinoma."}
        results = search_documents("what cancer did the patient have", documents=docs, top_k=5)
        assert len(results) >= 1
        assert "carcinoma" in results[0]["snippet"].lower()


class TestTabularTools:
    def test_compute_table_stat_mean(self):
        import pandas as pd

        from mosaicx.query.tools import compute_table_stat

        data = {
            "cohort.csv": pd.DataFrame(
                {"BMI": [20.0, 25.0, 30.0], "Sex": ["F", "M", "F"]}
            )
        }
        rows = compute_table_stat(
            "cohort.csv",
            data=data,
            column="BMI",
            operation="mean",
        )
        assert len(rows) == 1
        assert rows[0]["operation"] == "mean"
        assert rows[0]["column"] == "BMI"
        assert rows[0]["value"] == "25"
        assert rows[0]["backend"] in {"duckdb", "pandas"}

    def test_compute_table_stat_where_clause(self):
        import pandas as pd

        from mosaicx.query.tools import compute_table_stat

        data = {
            "cohort.csv": pd.DataFrame(
                {"BMI": [20.0, 25.0, 30.0], "Sex": ["F", "M", "F"]}
            )
        }
        rows = compute_table_stat(
            "cohort.csv",
            data=data,
            column="BMI",
            operation="mean",
            where="BMI >= 25",
        )
        assert len(rows) == 1
        assert rows[0]["value"] == "27.5"
        assert rows[0]["row_count"] == 2
        assert rows[0]["backend"] in {"duckdb", "pandas"}

    def test_analyze_table_question_derives_mean_evidence(self):
        import pandas as pd

        from mosaicx.query.tools import analyze_table_question

        data = {
            "cohort.csv": pd.DataFrame(
                {"BMI": [20.0, 25.0, 30.0], "Age": [40, 55, 60]}
            )
        }
        hits = analyze_table_question(
            "what is the average BMI of the cohort?",
            data=data,
            top_k=3,
        )
        assert len(hits) >= 1
        assert hits[0]["evidence_type"] == "table_stat"
        assert "mean of BMI" in hits[0]["snippet"]
        assert "25" in hits[0]["snippet"]
        assert "engine=" in hits[0]["snippet"]

    def test_search_tables_returns_row_level_snippets(self):
        import pandas as pd

        from mosaicx.query.tools import search_tables

        data = {
            "cohort.csv": pd.DataFrame(
                {"Subject": ["S1", "S2"], "BMI": [21.5, 26.1], "Sex": ["F", "M"]}
            )
        }
        hits = search_tables("find bmi rows", data=data, top_k=2)
        assert len(hits) >= 1
        assert hits[0]["evidence_type"] == "table_row"
        assert "row" in hits[0]["snippet"].lower()
        assert "BMI=" in hits[0]["snippet"]

    def test_run_table_sql_returns_rows(self):
        import pandas as pd

        duckdb = pytest.importorskip("duckdb")
        assert duckdb is not None

        from mosaicx.query.tools import run_table_sql

        data = {
            "cohort.csv": pd.DataFrame(
                {"BMI": [20.0, 25.0, 30.0], "Sex": ["F", "M", "F"]}
            )
        }
        rows = run_table_sql(
            "cohort.csv",
            data=data,
            sql="SELECT AVG(BMI) AS mean_bmi FROM _mosaicx_table",
        )
        assert len(rows) == 1
        assert rows[0]["mean_bmi"] == "25"


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

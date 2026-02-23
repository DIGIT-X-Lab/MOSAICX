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

    def test_build_document_chunks_splits_long_documents(self):
        from mosaicx.query.tools import build_document_chunks

        long_text = ("intro " * 480) + "Study Date: 2025-09-10. Key finding: lesion grew to 16 mm."
        docs = {"long_report.txt": long_text}
        chunk_index = build_document_chunks(
            docs,
            chunk_chars=700,
            overlap_chars=120,
            max_chunks_per_document=50,
        )
        assert "long_report.txt" in chunk_index
        assert len(chunk_index["long_report.txt"]) >= 3
        assert chunk_index["long_report.txt"][0]["chunk_id"] == 0

    def test_search_document_chunks_finds_deep_match(self):
        from mosaicx.query.tools import build_document_chunks
        from mosaicx.query.tools import search_document_chunks

        long_text = ("intro " * 480) + "Study Date: 2025-09-10. Key finding: lesion grew to 16 mm."
        docs = {"long_report.txt": long_text}
        chunk_index = build_document_chunks(docs, chunk_chars=700, overlap_chars=120)
        results = search_document_chunks(
            "lesion grew 16 mm on 2025-09-10",
            documents=docs,
            top_k=3,
            chunk_index=chunk_index,
        )
        assert len(results) >= 1
        assert results[0]["evidence_type"] == "text_chunk"
        assert "16 mm" in results[0]["snippet"]

    def test_get_document_chunk_by_id_and_missing(self):
        from mosaicx.query.tools import build_document_chunks
        from mosaicx.query.tools import get_document_chunk
        from mosaicx.query.tools import search_document_chunks

        long_text = ("intro " * 480) + "Study Date: 2025-09-10. Key finding: lesion grew to 16 mm."
        docs = {"long_report.txt": long_text}
        chunk_index = build_document_chunks(docs, chunk_chars=700, overlap_chars=120)
        hits = search_document_chunks(
            "lesion grew 16 mm",
            documents=docs,
            top_k=1,
            chunk_index=chunk_index,
        )
        assert hits
        chunk = get_document_chunk(
            "long_report.txt",
            int(hits[0]["chunk_id"]),
            chunk_index=chunk_index,
        )
        assert "16 mm" in chunk["text"]

        missing = get_document_chunk("long_report.txt", 99999, chunk_index=chunk_index)
        assert missing["error"] == "chunk_not_found"


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

    def test_analyze_table_question_subject_count_uses_distinct(self):
        import pandas as pd

        from mosaicx.query.tools import analyze_table_question

        data = {
            "cohort.csv": pd.DataFrame(
                {"Subject": ["S1", "S1", "S2", "S3"], "BMI": [20.0, 21.0, 25.0, 30.0]}
            )
        }
        hits = analyze_table_question(
            "how many subjects are there?",
            data=data,
            top_k=3,
        )
        assert len(hits) >= 1
        assert hits[0]["evidence_type"] == "table_stat"
        assert hits[0]["operation"] == "nunique"
        assert hits[0]["column"] == "Subject"
        assert hits[0]["value"] == 3
        assert "unique_count of Subject" in hits[0]["snippet"]

    def test_analyze_table_question_row_count_uses_table_rows(self):
        import pandas as pd

        from mosaicx.query.tools import analyze_table_question

        data = {
            "cohort.csv": pd.DataFrame(
                {"Subject": ["S1", "S1", "S2", "S3"], "BMI": [20.0, 21.0, 25.0, 30.0]}
            )
        }
        hits = analyze_table_question(
            "how many rows are in this dataset?",
            data=data,
            top_k=3,
        )
        assert len(hits) >= 1
        assert hits[0]["evidence_type"] == "table_stat"
        assert hits[0]["operation"] == "row_count"
        assert hits[0]["column"] == "__rows__"
        assert hits[0]["value"] == 4

    def test_analyze_table_question_ethnicity_count_uses_distinct(self):
        import pandas as pd

        from mosaicx.query.tools import analyze_table_question

        data = {
            "cohort.csv": pd.DataFrame(
                {"Ethnicity": ["Japanese", "Japanese", "German"], "Age": [50, 52, 49]}
            )
        }
        hits = analyze_table_question(
            "how many ethnicities are there?",
            data=data,
            top_k=4,
        )
        assert len(hits) >= 1
        top = hits[0]
        assert top["evidence_type"] == "table_stat"
        assert top["operation"] == "nunique"
        assert top["column"] == "Ethnicity"
        assert top["value"] == 2

    def test_analyze_table_question_distinct_values_for_followup(self):
        import pandas as pd

        from mosaicx.query.tools import analyze_table_question

        data = {
            "cohort.csv": pd.DataFrame(
                {"Ethnicity": ["Japanese", "Japanese", "German"], "Age": [50, 52, 49]}
            )
        }
        hits = analyze_table_question(
            "what are they? (follow-up context: how many ethnicities are there?)",
            data=data,
            top_k=6,
        )
        assert len(hits) >= 1
        assert any(h.get("evidence_type") == "table_value" for h in hits)
        assert any(str(h.get("value")) == "Japanese" for h in hits)

    def test_analyze_table_question_count_plus_enumeration_phrase_returns_values(self):
        import pandas as pd

        from mosaicx.query.tools import analyze_table_question

        data = {
            "cohort.csv": pd.DataFrame(
                {"Ethnicity": ["Japanese", "Japanese", "German"], "Age": [50, 52, 49]}
            )
        }
        hits = analyze_table_question(
            "how many ethnicities are there and what are there?",
            data=data,
            top_k=6,
        )
        assert len(hits) >= 1
        assert any(h.get("evidence_type") == "table_value" for h in hits)
        assert any(str(h.get("value")) == "Japanese" for h in hits)
        assert any(str(h.get("value")) == "German" for h in hits)

    def test_analyze_table_question_distribution_returns_values_and_count(self):
        import pandas as pd

        from mosaicx.query.tools import analyze_table_question

        data = {
            "cohort.csv": pd.DataFrame(
                {"Sex": ["M", "F", "M", "M", "F"], "Age": [50, 52, 49, 61, 47]}
            )
        }
        hits = analyze_table_question(
            "what is the distribution of male and female in the cohort?",
            data=data,
            top_k=8,
        )
        assert len(hits) >= 1
        assert any(h.get("evidence_type") == "table_value" for h in hits)
        assert any(h.get("evidence_type") == "table_stat" for h in hits)
        assert any(str(h.get("value")) in {"M", "F"} for h in hits if h.get("evidence_type") == "table_value")

    def test_list_distinct_values_returns_counts(self):
        import pandas as pd

        from mosaicx.query.tools import list_distinct_values

        data = {
            "cohort.csv": pd.DataFrame(
                {"Ethnicity": ["Japanese", "Japanese", "German"]}
            )
        }
        rows = list_distinct_values(
            "cohort.csv",
            data=data,
            column="Ethnicity",
            limit=10,
        )
        assert len(rows) == 2
        values = {r["value"]: r["count"] for r in rows}
        assert values["Japanese"] == 2
        assert values["German"] == 1

    def test_infer_table_roles_detects_numeric_id_and_text(self):
        import pandas as pd

        from mosaicx.query.tools import infer_table_roles

        data = {
            "cohort.csv": pd.DataFrame(
                {
                    "SubjectID": ["S1", "S2", "S3", "S4"],
                    "BMI": [20.0, 25.0, 30.0, 28.0],
                    "Notes": [
                        "Known prostate carcinoma with stable osseous lesion in lumbar spine.",
                        "Follow-up exam with no suspicious visceral metastasis identified.",
                        "Lymph node remains mildly enlarged but otherwise disease burden stable.",
                        "No focal liver lesion, no pleural effusion, unchanged imaging pattern.",
                    ],
                }
            )
        }
        profile = infer_table_roles("cohort.csv", data=data)
        assert "BMI" in profile["roles"]["numeric"]
        assert "SubjectID" in profile["roles"]["id"]
        assert profile["column_roles"]["Notes"] == "text"

    def test_suggest_table_columns_handles_entity_count_without_exact_column_match(self):
        import pandas as pd

        from mosaicx.query.tools import suggest_table_columns

        data = {
            "cohort.csv": pd.DataFrame(
                {
                    "patient_id": ["P1", "P2", "P3", "P4"],
                    "BMI": [21.0, 24.0, 31.0, 29.0],
                }
            )
        }
        hits = suggest_table_columns(
            "how many participants are included?",
            data=data,
            top_k=3,
        )
        assert len(hits) >= 1
        assert hits[0]["column"] == "patient_id"
        assert hits[0]["evidence_type"] == "table_column"

    def test_suggest_table_columns_maps_gender_to_sex(self):
        import pandas as pd

        from mosaicx.query.tools import suggest_table_columns

        data = {
            "cohort.csv": pd.DataFrame(
                {
                    "Subject": ["S1", "S2", "S3"],
                    "Sex": ["M", "F", "M"],
                    "Age": [50, 54, 47],
                }
            )
        }
        hits = suggest_table_columns(
            "how many genders are there?",
            data=data,
            top_k=3,
        )
        assert len(hits) >= 1
        assert hits[0]["column"] == "Sex"

    def test_analyze_table_question_maps_male_female_to_sex_distinct(self):
        import pandas as pd

        from mosaicx.query.tools import analyze_table_question

        data = {
            "cohort.csv": pd.DataFrame(
                {
                    "Subject": ["S1", "S2", "S3", "S4"],
                    "Sex": ["M", "F", "M", "F"],
                }
            )
        }
        hits = analyze_table_question(
            "how many male and female?",
            data=data,
            top_k=5,
        )
        assert len(hits) >= 1
        table_stats = [h for h in hits if h.get("evidence_type") == "table_stat"]
        assert table_stats
        assert any(h.get("column") == "Sex" and h.get("operation") == "nunique" for h in table_stats)

    def test_profile_table_includes_numeric_summary_and_top_values(self):
        import pandas as pd

        from mosaicx.query.tools import profile_table

        data = {
            "cohort.csv": pd.DataFrame(
                {
                    "BMI": [20.0, 25.0, 30.0, 30.0],
                    "Sex": ["F", "M", "F", "F"],
                }
            )
        }
        profile = profile_table("cohort.csv", data=data, max_columns=10, top_values=3)
        assert profile["rows"] == 4
        by_name = {c["name"]: c for c in profile["columns"]}
        assert "summary" in by_name["BMI"]
        assert by_name["BMI"]["summary"]["mean"] == "26.25"
        assert "top_values" in by_name["Sex"]
        assert by_name["Sex"]["top_values"][0]["value"] == "F"

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

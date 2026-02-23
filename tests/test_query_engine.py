# tests/test_query_engine.py
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestQueryEngineStructure:
    def test_engine_has_ask_method(self):
        """QueryEngine must expose an ask() method."""
        from mosaicx.query.engine import QueryEngine

        assert hasattr(QueryEngine, "ask")

    def test_engine_requires_session(self):
        """QueryEngine must require a session argument."""
        from mosaicx.query.engine import QueryEngine

        with pytest.raises(TypeError):
            QueryEngine()  # must provide session

    def test_engine_accepts_session(self, tmp_path: Path):
        """QueryEngine can be constructed with a valid session."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Patient has cough.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        assert engine.session is session

    def test_engine_rejects_closed_session(self, tmp_path: Path):
        """QueryEngine raises ValueError if session is already closed."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Patient has cough.")
        session = QuerySession(sources=[f])
        session.close()
        with pytest.raises(ValueError, match="closed"):
            QueryEngine(session=session)


class TestQueryEngineDocuments:
    def test_prepares_text_documents(self, tmp_path: Path):
        """Engine prepares text representations of loaded data for tools."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Patient has 5mm nodule.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        docs = engine.documents
        assert "report.txt" in docs
        assert "5mm nodule" in docs["report.txt"]

    def test_prepares_json_documents(self, tmp_path: Path):
        """Engine converts JSON data to text representation."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "value"}))
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        docs = engine.documents
        assert "data.json" in docs
        # JSON data should be stringified
        assert "key" in docs["data.json"]

    def test_prepares_dataframe_documents(self, tmp_path: Path):
        """Engine converts DataFrame data to text representation."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        docs = engine.documents
        assert "data.csv" in docs
        assert "Alice" in docs["data.csv"]


class TestQueryEngineConversation:
    def test_ask_raises_on_closed_session(self, tmp_path: Path):
        """ask() raises ValueError if session was closed after engine init."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Normal exam.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        session.close()
        with pytest.raises(ValueError, match="closed"):
            engine.ask("What does the report say?")

    def test_ask_requires_nonempty_question(self, tmp_path: Path):
        """ask() raises ValueError for empty questions."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Normal exam.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        with pytest.raises(ValueError, match="question"):
            engine.ask("")
        with pytest.raises(ValueError, match="question"):
            engine.ask("   ")

    def test_ask_structured_returns_grounded_payload(self, tmp_path: Path, monkeypatch):
        """ask_structured should return answer + citations + confidence."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Patient has a 5mm nodule in the right upper lobe.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        monkeypatch.setattr(
            engine,
            "_run_query_once",
            lambda q: ("The report notes a 5mm nodule in the right upper lobe.", [
                {"source": "report.txt", "snippet": "5mm nodule in the right upper lobe", "score": 3},
            ]),
        )

        payload = engine.ask_structured("What is the nodule size?")
        assert payload["answer"]
        assert isinstance(payload["citations"], list)
        assert len(payload["citations"]) >= 1
        assert payload["citations"][0]["source"] == "report.txt"
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["fallback_used"] is False
        assert payload["rescue_used"] is False
        # user + assistant turn
        assert len(session.conversation) == 2

    def test_ask_structured_falls_back_when_llm_fails(self, tmp_path: Path, monkeypatch):
        """ask_structured should still return grounded output if RLM call fails."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Patient has a 5mm nodule in the right upper lobe.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        def _raise(_q):
            raise RuntimeError("LM offline")

        monkeypatch.setattr(engine, "_run_query_once", _raise)
        payload = engine.ask_structured("What is the nodule size?")
        assert payload["answer"]
        assert payload["fallback_used"] is True
        assert "RuntimeError" in str(payload["fallback_reason"])
        assert "rescue_used" in payload
        assert isinstance(payload["citations"], list)
        assert len(session.conversation) == 2

    def test_ask_structured_rescues_non_answer_when_evidence_exists(self, tmp_path: Path, monkeypatch):
        """If RLM returns a non-answer but citations exist, engine should rescue."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Study Date: 2025-08-01. Findings: No suspicious nodules.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        monkeypatch.setattr(
            engine,
            "_run_query_once",
            lambda q: (
                "I’m unable to retrieve the report contents.",
                [{"source": "report.txt", "snippet": "Study Date: 2025-08-01.", "score": 2}],
            ),
        )
        monkeypatch.setattr(
            engine,
            "_rescue_answer_with_evidence",
            lambda **kwargs: "Timeline from grounded evidence:\n- 2025-08-01 | report.txt: No suspicious nodules.",
        )

        payload = engine.ask_structured("Summarize with timeline")
        assert payload["rescue_used"] is True
        assert payload["rescue_reason"] == "non_answer_with_evidence"
        assert "Timeline from grounded evidence" in payload["answer"]

    def test_ask_structured_deterministic_tabular_count_and_values(self, tmp_path: Path):
        """Tabular count+value queries should resolve deterministically without LM dependency."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text("Subject,Ethnicity,Sex\nS1,Japanese,M\nS2,Japanese,F\nS3,German,M\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        payload = engine.ask_structured("how many ethnicities are there and what are they?")
        answer = str(payload["answer"])
        assert "2 distinct Ethnicity values" in answer
        assert "Japanese" in answer
        assert "German" in answer
        assert any(c.get("evidence_type") == "table_stat" for c in payload["citations"])
        assert any(c.get("evidence_type") == "table_value" for c in payload["citations"])

    def test_ask_structured_schema_question_returns_all_columns(self, tmp_path: Path, monkeypatch):
        """Schema questions should return full column list and skip reconciler rewrites."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text("Subject,Ethnicity,Sex,Age,Weight,BMI,Injected_Activity\nS1,Japanese,M,50,80,25,300\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        def _must_not_run(**kwargs):
            raise AssertionError("reconciler should not run for schema deterministic answers")

        monkeypatch.setattr(engine, "_reconcile_answer_with_evidence", _must_not_run)

        payload = engine.ask_structured("what are the column names?")
        answer = str(payload["answer"])
        assert "Subject" in answer
        assert "Ethnicity" in answer
        assert "Injected_Activity" in answer
        assert payload["deterministic_used"] is True
        assert payload.get("deterministic_intent") == "schema"
        assert any(c.get("evidence_type") == "table_schema" for c in payload["citations"])

    def test_ask_structured_followup_all_column_names_uses_state(self, tmp_path: Path):
        """Follow-up schema request should keep table context and return full list."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text("Subject,Ethnicity,Sex,Age\nS1,Japanese,M,50\nS2,German,F,52\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        first = engine.ask_structured("what are the column names")
        second = engine.ask_structured("all of the column names please")

        assert "Subject" in str(first["answer"])
        assert "Ethnicity" in str(second["answer"])
        assert "Age" in str(second["answer"])
        assert second.get("deterministic_intent") == "schema"

    def test_ask_structured_count_values_accepts_table_value_computed_evidence(self, tmp_path: Path, monkeypatch):
        """For count+values questions, value-count evidence should satisfy computed guard."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text("Subject,Ethnicity\nS1,Japanese\nS2,Japanese\nS3,German\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        hits = [
            {
                "source": "cohort.csv",
                "snippet": "Distinct Ethnicity: Japanese (count=2, engine=duckdb)",
                "score": 90,
                "evidence_type": "table_value",
                "operation": "distinct_values",
                "column": "Ethnicity",
                "value": "Japanese",
                "count": 2,
                "backend": "duckdb",
            },
            {
                "source": "cohort.csv",
                "snippet": "Distinct Ethnicity: German (count=1, engine=duckdb)",
                "score": 90,
                "evidence_type": "table_value",
                "operation": "distinct_values",
                "column": "Ethnicity",
                "value": "German",
                "count": 1,
                "backend": "duckdb",
            },
        ]
        monkeypatch.setattr(engine, "_run_query_once", lambda q: ("There are 2 ethnicities: Japanese, German.", hits))
        monkeypatch.setattr(engine, "_build_citations", lambda **kwargs: hits)

        payload = engine.ask_structured("how many ethnicities are there and what are they?")
        assert "reliable numeric answer" not in str(payload["answer"])
        assert payload.get("rescue_reason") != "missing_computed_evidence"

    def test_count_values_after_schema_turns_uses_topic_not_prior_column_dump(self, tmp_path: Path):
        """Topic-bearing count+values prompts should not be derailed by prior schema turns."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text(
            "Subject,Ethnicity,Sex,Injected_Activity\n"
            "S1,Japanese,M,300\n"
            "S2,Japanese,F,280\n"
            "S3,German,M,305\n"
        )
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        _ = engine.ask_structured("what are the column names?")
        _ = engine.ask_structured("all of the column names please")
        payload = engine.ask_structured("how many ethnicities are there and what are they?")

        answer = str(payload["answer"])
        assert "Ethnicity" in answer
        assert "Japanese" in answer
        assert "German" in answer
        assert "Injected_Activity" not in answer
        assert payload.get("rescue_reason") != "missing_computed_evidence"

    def test_ask_structured_deterministic_aggregate_for_average(self, tmp_path: Path):
        """Average-style questions should use deterministic aggregate execution when possible."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text("Subject,BMI\nS1,20\nS2,25\nS3,30\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        payload = engine.ask_structured("what is the average BMI?")
        assert payload["deterministic_used"] is True
        assert payload.get("deterministic_intent") == "aggregate"
        assert "mean of BMI" in " ".join(str(c.get("snippet", "")) for c in payload["citations"])

    def test_analytic_guard_not_applied_for_text_only_sessions(self, tmp_path: Path, monkeypatch):
        """Numeric fail-closed guard should not trigger for text-only sources."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Two reports were reviewed.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        seed_hits = [
            {
                "source": "report.txt",
                "snippet": "Two reports were reviewed.",
                "score": 3,
                "evidence_type": "text",
            }
        ]
        monkeypatch.setattr(
            engine,
            "_run_query_once",
            lambda q: ("There were two reports.", seed_hits),
        )
        monkeypatch.setattr(
            engine,
            "_build_citations",
            lambda **kwargs: seed_hits,
        )

        payload = engine.ask_structured("how many reports were reviewed?")
        assert payload["answer"] == "There were two reports."
        assert payload["rescue_reason"] != "missing_computed_evidence"

    def test_analytic_guard_fail_closed_for_tabular_without_computed_evidence(self, tmp_path: Path, monkeypatch):
        """Tabular analytics should fail closed when no computed citation is available."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text("BMI\n20\n25\n30\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        monkeypatch.setattr(
            engine,
            "_run_query_once",
            lambda q: ("95th percentile BMI is 30.", []),
        )
        monkeypatch.setattr(
            engine,
            "_build_citations",
            lambda **kwargs: [],
        )

        payload = engine.ask_structured("what is the average bmi?")
        assert payload["rescue_used"] is True
        assert payload["rescue_reason"] == "missing_computed_evidence"
        assert "reliable numeric answer" in payload["answer"]

    def test_analytic_guard_uses_programmatic_sql_before_fail_closed(self, tmp_path: Path, monkeypatch):
        """If deterministic evidence is missing, SQL planner rescue should run before fail-closed."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text("BMI\n20\n25\n30\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        monkeypatch.setattr(
            engine,
            "_run_query_once",
            lambda q: ("Average BMI is 30.", []),
        )
        monkeypatch.setattr(
            engine,
            "_build_citations",
            lambda **kwargs: kwargs.get("seed_hits", []),
        )
        monkeypatch.setattr(
            engine,
            "_attempt_programmatic_sql_answer",
            lambda _q: (
                "mean_bmi: 25.",
                [
                    {
                        "source": "cohort.csv",
                        "snippet": "Computed SQL on cohort.csv: SELECT AVG(BMI) AS mean_bmi FROM _mosaicx_table -> [{\\\"mean_bmi\\\": \\\"25\\\"}]",
                        "score": 92,
                        "evidence_type": "table_sql",
                    }
                ],
                "sql_analytic",
            ),
        )

        payload = engine.ask_structured("what is the 95th percentile bmi?")
        assert payload["rescue_used"] is True
        assert payload["rescue_reason"] == "programmatic_sql"
        assert payload["deterministic_intent"] == "sql_analytic"
        assert "mean_bmi: 25" in payload["answer"]
        assert payload["rescue_reason"] != "missing_computed_evidence"

    def test_adapter_parse_error_retries_once_before_fallback(self, tmp_path: Path, monkeypatch):
        """Adapter parse errors should retry once with stricter plain-text guidance."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Patient has a 5mm nodule in the right upper lobe.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        calls = {"n": 0}

        def _flaky(_q):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("AdapterParseError: LM response cannot be serialized to a JSON object.")
            return (
                "The report notes a 5mm nodule in the right upper lobe.",
                [{"source": "report.txt", "snippet": "5mm nodule in the right upper lobe", "score": 4}],
            )

        monkeypatch.setattr(engine, "_run_query_once", _flaky)

        payload = engine.ask_structured("What is the nodule size?")
        assert calls["n"] == 2
        assert payload["fallback_used"] is False
        assert payload["fallback_code"] is None
        assert "5mm nodule" in payload["answer"]

    def test_reconciler_corrects_unsupported_draft_answer(self, tmp_path: Path, monkeypatch):
        """Evidence reconciler should correct unsupported draft answers."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Lymph nodes: Right external iliac node increased in size (now 16 mm short-axis).")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        monkeypatch.setattr(
            engine,
            "_run_query_once",
            lambda q: (
                "Size change: 0 mm (no change).",
                [
                    {"source": "P001_CT_2025-08-01.pdf", "snippet": "Right external iliac node measured 12 mm.", "score": 3},
                    {"source": "P001_CT_2025-09-10.pdf", "snippet": "Right external iliac node increased in size (now 16 mm short-axis).", "score": 4},
                ],
            ),
        )
        monkeypatch.setattr(
            engine,
            "_build_citations",
            lambda **kwargs: kwargs["seed_hits"],
        )
        monkeypatch.setattr(
            engine,
            "_reconcile_answer_with_evidence",
            lambda **kwargs: "Lesion size increased from 12 mm to 16 mm (+4 mm).",
        )

        payload = engine.ask_structured("how much did the lesion size change")
        assert payload["rescue_used"] is True
        assert payload["rescue_reason"] == "evidence_reconciler"
        assert "12 mm to 16 mm" in payload["answer"]

    def test_non_answer_marker_not_specified(self, tmp_path: Path):
        """Common fallback phrasing should trigger non-answer detection."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Indication: Known prostate carcinoma.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        assert engine._looks_like_non_answer("The cancer type is not specified.")

    def test_confidence_high_for_short_modality_answers(self, tmp_path: Path):
        """Grounding should be high when concise answer is explicitly present in evidence."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Modality: CT")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        citations = [
            {"source": "P001_CT_2025-08-01.pdf", "snippet": "Modality: CT", "score": 1, "rank": 24},
            {"source": "P001_CT_2025-09-10.pdf", "snippet": "Modality: CT", "score": 1, "rank": 24},
        ]
        conf = engine._citation_confidence(
            "what kind of imaging modality was used in this patient throughout",
            "CT",
            citations,
        )
        assert conf >= 0.75

    def test_confidence_high_for_scan_dates(self, tmp_path: Path):
        """Date-list answers should score high when both dates are cited."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Study Date: 2025-08-01")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        citations = [
            {"source": "P001_CT_2025-08-01.pdf", "snippet": "Study Date: 2025-08-01", "score": 2, "rank": 24},
            {"source": "P001_CT_2025-09-10.pdf", "snippet": "Study Date: 2025-09-10", "score": 2, "rank": 24},
        ]
        conf = engine._citation_confidence(
            "what were the imaging scan dates",
            "- 2025-08-01\n- 2025-09-10",
            citations,
        )
        assert conf >= 0.75

    def test_confidence_drops_for_answer_not_in_evidence(self, tmp_path: Path):
        """Grounding should decrease when answer values are missing from citations."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Modality: CT")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        citations = [
            {"source": "P001_CT_2025-08-01.pdf", "snippet": "Modality: CT", "score": 1, "rank": 24},
            {"source": "P001_CT_2025-09-10.pdf", "snippet": "Modality: CT", "score": 1, "rank": 24},
        ]
        conf = engine._citation_confidence(
            "what kind of imaging modality was used in this patient throughout",
            "MRI",
            citations,
        )
        assert conf < 0.6

    def test_build_citations_filters_generic_delta_noise(self, tmp_path: Path, monkeypatch):
        """Delta citation ranking should suppress generic headers when measured lines exist."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("dummy")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        seed_hits = [
            {"source": "P001_CT_2025-09-10.pdf", "snippet": "Radiology Report", "score": 20},
            {"source": "P001_CT_2025-09-10.pdf", "snippet": "Lymph nodes: Right external iliac node increased in size (now 16 mm short-axis).", "score": 5},
            {"source": "P001_CT_2025-08-01.pdf", "snippet": "Right external iliac node measured 12 mm short-axis.", "score": 4},
        ]

        monkeypatch.setattr("mosaicx.query.tools.search_documents", lambda *args, **kwargs: [])
        citations = engine._build_citations(
            question="how much did the lesion size change",
            answer="",
            seed_hits=seed_hits,
            top_k=3,
        )

        assert len(citations) >= 2
        assert "16 mm" in citations[0]["snippet"] or "16 mm" in citations[1]["snippet"]
        assert all(c["snippet"].lower() != "radiology report" for c in citations)

    def test_build_citations_adds_tabular_computed_evidence(self, tmp_path: Path):
        """Citations should include computed table evidence for cohort-stat questions."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "cohort.csv"
        f.write_text("BMI,Age\n20,40\n25,55\n30,60\n")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)

        citations = engine._build_citations(
            question="what is the average bmi of the cohort?",
            answer="Average BMI is approximately 25.",
            seed_hits=[],
            top_k=3,
        )
        assert len(citations) >= 1
        assert any(c.get("evidence_type") == "table_stat" for c in citations)
        assert any("mean of BMI" in c["snippet"] for c in citations)


class TestQueryEngineConfig:
    def test_engine_has_configurable_max_iterations(self, tmp_path: Path):
        """QueryEngine accepts max_iterations parameter."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Normal exam.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session, max_iterations=5)
        assert engine.max_iterations == 5

    def test_engine_default_max_iterations(self, tmp_path: Path):
        """QueryEngine defaults to 20 max_iterations."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Normal exam.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        assert engine.max_iterations == 20

    def test_engine_has_configurable_verbose(self, tmp_path: Path):
        """QueryEngine accepts verbose parameter."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Normal exam.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session, verbose=True)
        assert engine.verbose is True


class TestQueryEngineImport:
    def test_module_importable_without_dspy_configured(self):
        """Module imports without DSPy being fully configured."""
        from mosaicx.query import engine

        assert hasattr(engine, "QueryEngine")


@pytest.fixture()
def _configure_dspy_for_test():
    """Configure DSPy from MosaicxConfig for integration tests."""
    import dspy

    if dspy.settings.lm is not None:
        yield
        return

    from mosaicx.config import get_config
    from mosaicx.metrics import make_harmony_lm

    cfg = get_config()
    if not cfg.api_key or not cfg.lm:
        pytest.skip("No MOSAICX_API_KEY/LM configured")
    lm = make_harmony_lm(cfg.lm, api_key=cfg.api_key, api_base=cfg.api_base, temperature=cfg.lm_temperature)
    dspy.configure(lm=lm)
    yield


@pytest.mark.integration
class TestQueryEngineIntegration:
    """Integration tests requiring a running LLM.

    These tests auto-configure DSPy from MosaicxConfig via fixture and are
    skipped only when MOSAICX_API_KEY/LM config is missing. Run manually:
        MOSAICX_API_KEY=... pytest tests/test_query_engine.py -m integration
    """

    def test_ask_returns_string(self, tmp_path: Path, _configure_dspy_for_test):
        """Full ask() round-trip returns a string answer."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Patient has 5mm nodule in right upper lobe.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        answer = engine.ask("What size is the nodule?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_ask_tracks_conversation(self, tmp_path: Path, _configure_dspy_for_test):
        """ask() appends user and assistant turns to conversation."""
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("Patient has 5mm nodule in right upper lobe.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        engine.ask("What does the report say?")
        conv = session.conversation
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"

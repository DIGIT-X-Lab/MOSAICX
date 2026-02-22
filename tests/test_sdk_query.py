"""Behavioral tests for the SDK query() function."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSDKQuerySessionCreation:
    """SDK query() should create functional QuerySessions."""

    def test_query_returns_session(self, tmp_path):
        from mosaicx.query.session import QuerySession
        from mosaicx.sdk import query

        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\n")
        session = query(sources=[f])
        assert isinstance(session, QuerySession)
        session.close()

    def test_session_loads_text_data(self, tmp_path):
        from mosaicx.sdk import query

        f = tmp_path / "report.txt"
        f.write_text("Patient has 5mm nodule in the RUL.")
        session = query(sources=[f])
        assert "report.txt" in session.data
        assert "5mm nodule" in session.data["report.txt"]
        session.close()

    def test_session_loads_json_data(self, tmp_path):
        from mosaicx.sdk import query

        f = tmp_path / "extraction.json"
        f.write_text(json.dumps({"findings": [{"anatomy": "RUL"}]}))
        session = query(sources=[f])
        assert "extraction.json" in session.data
        data = session.data["extraction.json"]
        assert isinstance(data, (dict, str))
        session.close()

    def test_session_loads_csv_data(self, tmp_path):
        from mosaicx.sdk import query

        f = tmp_path / "patients.csv"
        f.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n")
        session = query(sources=[f])
        assert "patients.csv" in session.data
        session.close()

    def test_session_loads_multiple_sources(self, tmp_path):
        from mosaicx.sdk import query

        f1 = tmp_path / "report.txt"
        f1.write_text("Normal exam.")
        f2 = tmp_path / "data.json"
        f2.write_text(json.dumps({"key": "value"}))
        session = query(sources=[f1, f2])
        assert len(session.data) == 2
        session.close()

    def test_session_catalog_has_metadata(self, tmp_path):
        from mosaicx.sdk import query

        f = tmp_path / "report.txt"
        f.write_text("Sample report text for testing.")
        session = query(sources=[f])
        assert len(session.catalog) == 1
        meta = session.catalog[0]
        assert meta.name == "report.txt"
        assert meta.size > 0
        session.close()


class TestSDKQuerySessionLifecycle:
    """Session lifecycle management."""

    def test_session_close(self, tmp_path):
        from mosaicx.sdk import query

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        session = query(sources=[f])
        assert not session.closed
        session.close()
        assert session.closed

    def test_closed_session_rejects_turns(self, tmp_path):
        from mosaicx.sdk import query

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        session = query(sources=[f])
        session.close()
        with pytest.raises(ValueError):
            session.add_turn("user", "hello")

    def test_conversation_tracking(self, tmp_path):
        from mosaicx.sdk import query

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        session = query(sources=[f])
        session.add_turn("user", "What is A?")
        session.add_turn("assistant", "A is 1.")
        assert len(session.conversation) == 2
        assert session.conversation[0]["role"] == "user"
        assert session.conversation[1]["role"] == "assistant"
        session.close()

    def test_session_ask_structured_returns_dict(self, tmp_path, monkeypatch):
        from mosaicx.sdk import query

        f = tmp_path / "data.txt"
        f.write_text("Patient has 5mm nodule.")
        session = query(sources=[f])

        fake_dspy = types.SimpleNamespace(
            settings=types.SimpleNamespace(lm=object())
        )
        monkeypatch.setitem(sys.modules, "dspy", fake_dspy)

        expected = {
            "question": "What is the finding?",
            "answer": "5mm nodule",
            "citations": [{"source": "data.txt", "snippet": "5mm nodule", "score": 2}],
            "confidence": 0.8,
            "sources_consulted": ["data.txt"],
            "turn_index": 1,
        }
        with patch("mosaicx.query.engine.QueryEngine.ask_structured", return_value=expected):
            out = session.ask_structured("What is the finding?")
        assert out["answer"] == "5mm nodule"
        assert isinstance(out["citations"], list)
        session.close()


class TestQueryEngine:
    """QueryEngine construction and validation."""

    def test_engine_rejects_closed_session(self, tmp_path):
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.txt"
        f.write_text("test")
        session = QuerySession(sources=[f])
        session.close()
        with pytest.raises(ValueError, match="closed"):
            QueryEngine(session=session)

    def test_engine_documents_include_source_content(self, tmp_path):
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("5mm nodule detected in RUL.")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        assert "report.txt" in engine.documents
        assert "5mm" in engine.documents["report.txt"]

    def test_engine_ask_requires_nonempty_question(self, tmp_path):
        from mosaicx.query.engine import QueryEngine
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.txt"
        f.write_text("test content")
        session = QuerySession(sources=[f])
        engine = QueryEngine(session=session)
        with pytest.raises(ValueError, match="question"):
            engine.ask("")
        with pytest.raises(ValueError, match="question"):
            engine.ask("   ")

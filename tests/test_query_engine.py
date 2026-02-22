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


@pytest.mark.integration
class TestQueryEngineIntegration:
    """Integration tests requiring a running LLM and Deno.

    These are skipped in CI; run manually with:
        pytest tests/test_query_engine.py -m integration
    """

    def test_ask_returns_string(self, tmp_path: Path):
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

    def test_ask_tracks_conversation(self, tmp_path: Path):
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

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
        assert isinstance(payload["citations"], list)
        assert len(session.conversation) == 2


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


def _dspy_lm_available() -> bool:
    """Check if a DSPy LM is already configured."""
    try:
        import dspy
        return dspy.settings.lm is not None
    except Exception:
        return False


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
@pytest.mark.skipif(not _dspy_lm_available(), reason="No DSPy LM configured (run with MOSAICX_API_KEY set)")
class TestQueryEngineIntegration:
    """Integration tests requiring a running LLM.

    These are skipped unless DSPy is configured with an LM. Run manually:
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

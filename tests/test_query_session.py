from __future__ import annotations

import json
from pathlib import Path


class TestQuerySession:
    def test_session_loads_sources(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "value"}))
        session = QuerySession(sources=[f])
        assert len(session.catalog) == 1
        assert session.catalog[0].name == "data.json"

    def test_session_catalog_metadata(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f1 = tmp_path / "report.txt"
        f1.write_text("Patient has cough.")
        f2 = tmp_path / "data.csv"
        f2.write_text("a,b\n1,2\n")
        session = QuerySession(sources=[f1, f2])
        assert len(session.catalog) == 2
        types = {m.source_type for m in session.catalog}
        assert "document" in types
        assert "dataframe" in types

    def test_session_conversation_starts_empty(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("text")
        session = QuerySession(sources=[f])
        assert session.conversation == []

    def test_session_close(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f = tmp_path / "report.txt"
        f.write_text("text")
        session = QuerySession(sources=[f])
        session.close()
        assert session.closed

    def test_session_data_access(self, tmp_path: Path):
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.json"
        f.write_text(json.dumps({"key": "value"}))
        session = QuerySession(sources=[f])
        assert "data.json" in session.data
        assert session.data["data.json"]["key"] == "value"

from __future__ import annotations

import pytest


class TestSDKQuery:
    def test_query_function_exists(self):
        from mosaicx.sdk import query

        assert callable(query)

    def test_query_returns_session(self, tmp_path):
        from mosaicx.sdk import query
        from mosaicx.query.session import QuerySession

        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        session = query(sources=[f])
        assert isinstance(session, QuerySession)
        session.close()

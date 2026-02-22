from __future__ import annotations

import json

import pytest


class TestMCPQueryTools:
    def test_query_start_tool_registered(self):
        """query_start should be a registered MCP tool."""
        from mosaicx.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "query_start" in tool_names

    def test_query_ask_tool_registered(self):
        """query_ask should be a registered MCP tool."""
        from mosaicx.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "query_ask" in tool_names

    def test_query_close_tool_registered(self):
        """query_close should be a registered MCP tool."""
        from mosaicx.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "query_close" in tool_names

    def test_query_start_creates_session(self):
        """query_start should create a session and return session_id + catalog."""
        from mosaicx.mcp_server import _sessions, query_start

        result = json.loads(query_start(source_texts={"report.txt": "Patient presents with chest pain."}))
        assert "session_id" in result
        assert "catalog" in result
        assert len(result["catalog"]) == 1
        assert result["catalog"][0]["name"] == "report.txt"

        # Session should be stored
        sid = result["session_id"]
        assert sid in _sessions
        assert not _sessions[sid].closed

        # Clean up
        _sessions[sid].close()
        del _sessions[sid]

    def test_query_start_multiple_sources(self):
        """query_start should handle multiple source texts."""
        from mosaicx.mcp_server import _sessions, query_start

        sources = {
            "report_a.txt": "Findings: normal chest.",
            "report_b.txt": "Impression: no acute disease.",
        }
        result = json.loads(query_start(source_texts=sources))
        assert len(result["catalog"]) == 2
        names = {c["name"] for c in result["catalog"]}
        assert names == {"report_a.txt", "report_b.txt"}

        # Clean up
        sid = result["session_id"]
        _sessions[sid].close()
        del _sessions[sid]

    def test_query_start_empty_sources_error(self):
        """query_start should return an error for empty source_texts."""
        from mosaicx.mcp_server import query_start

        result = json.loads(query_start(source_texts={}))
        assert "error" in result

    def test_query_close_removes_session(self):
        """query_close should close and remove the session."""
        from mosaicx.mcp_server import _sessions, query_close, query_start

        start_result = json.loads(query_start(source_texts={"doc.txt": "Some text."}))
        sid = start_result["session_id"]
        assert sid in _sessions

        close_result = json.loads(query_close(session_id=sid))
        assert close_result["status"] == "closed"
        assert sid not in _sessions

    def test_query_close_unknown_session_error(self):
        """query_close should return an error for an unknown session_id."""
        from mosaicx.mcp_server import query_close

        result = json.loads(query_close(session_id="nonexistent-id"))
        assert "error" in result

    def test_query_ask_unknown_session_error(self):
        """query_ask should return an error for an unknown session_id."""
        from mosaicx.mcp_server import query_ask

        result = json.loads(query_ask(session_id="nonexistent-id", question="Hello?"))
        assert "error" in result

from __future__ import annotations

import json


class TestMCPVerifyTools:
    def test_verify_output_tool_registered(self):
        """verify_output should be a registered MCP tool."""
        from mosaicx.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "verify_output" in tool_names

    def test_verify_output_with_extraction(self):
        """verify_output should verify an extraction against source text."""
        from mosaicx.mcp_server import verify_output

        extraction = json.dumps({
            "findings": [{"anatomy": "RUL", "description": "5mm nodule"}],
        })
        result = json.loads(verify_output(
            source_text="Findings: 5mm nodule in the right upper lobe.",
            extraction=extraction,
        ))
        assert "verdict" in result
        assert "confidence" in result

    def test_verify_output_with_claim(self):
        """verify_output should verify a single claim against source text."""
        from mosaicx.mcp_server import verify_output

        result = json.loads(verify_output(
            source_text="Patient has a 5mm nodule.",
            claim="5mm nodule present",
        ))
        assert "verdict" in result

    def test_verify_output_invalid_json_extraction(self):
        """verify_output should return an error for invalid JSON extraction."""
        from mosaicx.mcp_server import verify_output

        result = json.loads(verify_output(
            source_text="Some text.",
            extraction="not valid json{",
        ))
        assert "error" in result

    def test_verify_output_no_extraction_or_claim(self):
        """verify_output should return an error when neither extraction nor claim is given."""
        from mosaicx.mcp_server import verify_output

        result = json.loads(verify_output(
            source_text="Some text.",
        ))
        assert "error" in result

    def test_verify_output_with_level(self):
        """verify_output should accept a level parameter."""
        from mosaicx.mcp_server import verify_output

        result = json.loads(verify_output(
            source_text="Patient has a 5mm nodule.",
            claim="5mm nodule present",
            level="quick",
        ))
        assert "verdict" in result
        assert result.get("level") == "deterministic"

    def test_verify_output_invalid_level(self):
        """verify_output should return an error for an unknown verification level."""
        from mosaicx.mcp_server import verify_output

        result = json.loads(verify_output(
            source_text="Some text.",
            claim="some claim",
            level="invalid_level",
        ))
        assert "error" in result

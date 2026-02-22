# tests/test_envelope.py
"""Tests for the _mosaicx metadata envelope builder."""
from __future__ import annotations

import json
from datetime import datetime, timezone


class TestBuildEnvelope:
    """Test build_envelope() returns a well-formed metadata dict."""

    def test_returns_all_required_keys(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        expected_keys = {
            "version",
            "pipeline",
            "template",
            "template_version",
            "model",
            "model_temperature",
            "timestamp",
            "duration_s",
            "tokens",
            "provenance",
            "verification",
            "document",
        }
        assert set(env.keys()) == expected_keys

    def test_pipeline_name_preserved(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="pathology")
        assert env["pipeline"] == "pathology"

    def test_default_values(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        assert env["template"] is None
        assert env["template_version"] is None
        assert env["provenance"] is False
        assert env["verification"] is None
        assert env["document"] is None
        assert env["duration_s"] is None

    def test_default_tokens(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        assert env["tokens"] == {"input": 0, "output": 0}

    def test_document_metadata_preserved(self):
        from mosaicx.envelope import build_envelope

        doc_meta = {
            "filename": "report.pdf",
            "pages": 3,
            "ocr_engine": "surya",
        }
        env = build_envelope(pipeline="extraction", document=doc_meta)
        assert env["document"] == doc_meta

    def test_token_counts_preserved(self):
        from mosaicx.envelope import build_envelope

        tokens = {"input": 1500, "output": 800}
        env = build_envelope(pipeline="radiology", tokens=tokens)
        assert env["tokens"] == {"input": 1500, "output": 800}

    def test_json_serializable(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(
            pipeline="radiology",
            template="chest_ct",
            template_version="1.2",
            duration_s=3.14,
            tokens={"input": 100, "output": 50},
            document={"filename": "scan.pdf"},
            verification={"passed": True, "checks": 5},
        )
        # json.dumps should not raise
        serialized = json.dumps(env)
        assert isinstance(serialized, str)

    def test_timestamp_is_valid_iso(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        ts = env["timestamp"]
        # Should parse without error
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None  # must be timezone-aware

    def test_version_is_nonempty_string(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        assert isinstance(env["version"], str)
        assert len(env["version"]) > 0

    def test_template_and_version_passed(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(
            pipeline="radiology",
            template="brain_mri",
            template_version="2.0",
        )
        assert env["template"] == "brain_mri"
        assert env["template_version"] == "2.0"

    def test_duration_passed(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology", duration_s=12.5)
        assert env["duration_s"] == 12.5

    def test_provenance_override(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology", provenance=True)
        assert env["provenance"] is True

    def test_verification_dict_preserved(self):
        from mosaicx.envelope import build_envelope

        ver = {"status": "passed", "checks_run": 3, "issues": []}
        env = build_envelope(pipeline="radiology", verification=ver)
        assert env["verification"] == ver

    def test_model_is_string(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        assert isinstance(env["model"], str)

    def test_model_temperature_is_numeric(self):
        from mosaicx.envelope import build_envelope

        env = build_envelope(pipeline="radiology")
        assert isinstance(env["model_temperature"], (int, float))

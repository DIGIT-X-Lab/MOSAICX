# tests/test_signals.py
"""Tests for Signals API client and CLI integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mosaicx.signals import (
    SignalsAuthError,
    SignalsClient,
    SignalsError,
    SignalsRateLimitError,
    SignalsUpstreamError,
    SignalsValidationError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_EVAL_RESPONSE = {
    "evaluation_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2026-03-14T12:30:00Z",
    "trust_score": 62,
    "verdict": "review",
    "verdict_title": "Review Required",
    "verdict_explanation": "The AI report requires clinical review.",
    "trust_score_composition": {
        "clinical_correctness": 0.45,
        "safety_penalties": 0.10,
        "structured_review": 0.02,
        "semantic_similarity": 0.05,
    },
    "metrics": {
        "semantic_similarity_bertscore": {
            "score": 0.20,
            "max_score": 1,
            "description": "Semantic overlap between reports.",
            "clinical_relevance": "Low similarity.",
        }
    },
    "safety_signals": [
        {
            "type": "missing_finding",
            "severity": "critical",
            "generated_text": "No acute abnormality.",
            "reference_text": "Small pleural effusion.",
            "explanation": "Missed finding.",
        }
    ],
    "structured_review": {
        "clinical_correctness_score": 1,
        "completeness_score": 1,
        "recommendation_appropriateness_score": 5,
        "review_priority": "critical",
        "risk_assessment": "High risk.",
        "key_issues": ["Missing effusion."],
    },
    "discrepancies": [
        {
            "generated_statement": "No acute abnormality.",
            "severity": "critical",
            "reference_statement": "Small pleural effusion.",
            "issue_type": "Missing Finding",
            "section": "Findings",
        }
    ],
    "metadata": {
        "modality": "CR",
        "anatomy": "chest",
        "model_name": None,
        "model_version": None,
        "pii_detected": False,
    },
}


def _mock_response(status_code: int = 200, json_data: dict | None = None, text: str = ""):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.text = text
    resp.json.return_value = json_data or {}
    resp.headers = {}
    return resp


# ---------------------------------------------------------------------------
# SignalsClient unit tests
# ---------------------------------------------------------------------------


class TestSignalsClient:
    """Test the HTTP client with mocked responses."""

    def test_health_success(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(200, {"status": "ok", "service": "Signals API"})

        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.health()

        assert result["status"] == "ok"

    def test_evaluate_success(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(200, SAMPLE_EVAL_RESPONSE)

        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            result = client.evaluate(
                generated_report="AI text",
                reference_report="Ref text",
                metadata={"modality": "CT"},
            )

        assert result["trust_score"] == 62
        assert result["verdict"] == "review"

        # Verify request body
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["mode"] == "report"
        assert body["generated_report"] == "AI text"
        assert body["metadata"]["modality"] == "CT"

    def test_evaluate_no_metadata(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(200, SAMPLE_EVAL_RESPONSE)

        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            client.evaluate(
                generated_report="AI text",
                reference_report="Ref text",
            )

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "metadata" not in body

    def test_auth_error(self):
        client = SignalsClient(api_key="bad_key")
        mock_resp = _mock_response(
            401,
            {"error": {"code": "unauthorized", "message": "Invalid API key"}},
        )

        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(SignalsAuthError) as exc_info:
                client.evaluate("a", "b")

        assert exc_info.value.status == 401
        assert "Invalid API key" in str(exc_info.value)

    def test_validation_error(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(
            400,
            {"error": {"code": "invalid_request", "message": "generated_report is required"}},
        )

        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(SignalsValidationError):
                client.evaluate("", "b")

    def test_rate_limit_error(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(
            429,
            {"error": {"code": "rate_limit_exceeded", "message": "Too many requests"}},
        )
        mock_resp.headers = {"Retry-After": "60"}

        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(SignalsRateLimitError) as exc_info:
                client.evaluate("a", "b")

        assert exc_info.value.retry_after == 60

    def test_upstream_error_502(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(
            502,
            {"error": {"code": "upstream_error", "message": "AI service unavailable"}},
        )

        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(SignalsUpstreamError):
                client.evaluate("a", "b")

    def test_upstream_error_504(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(
            504,
            {"error": {"code": "gateway_timeout", "message": "Timed out"}},
        )

        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(SignalsUpstreamError):
                client.evaluate("a", "b")

    def test_generic_error(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(
            500,
            {"error": {"code": "internal_error", "message": "Something broke"}},
        )

        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(SignalsError) as exc_info:
                client.evaluate("a", "b")
            assert exc_info.value.status == 500

    def test_invalid_json_response(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.ok = False
        mock_resp.json.side_effect = ValueError("No JSON")

        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(SignalsError, match="Invalid response"):
                client.evaluate("a", "b")

    def test_status_endpoint(self):
        client = SignalsClient(api_key="sk_test_123")
        mock_resp = _mock_response(200, SAMPLE_EVAL_RESPONSE)

        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            result = client.status("550e8400-e29b-41d4-a716-446655440000")

        assert result["evaluation_id"] == "550e8400-e29b-41d4-a716-446655440000"
        call_url = mock_get.call_args[0][0]
        assert "/status/550e8400" in call_url

    def test_custom_base_url(self):
        client = SignalsClient(api_key="sk_test_123", base_url="https://custom.api/v1/")
        assert client._base_url == "https://custom.api/v1"  # trailing slash stripped


# ---------------------------------------------------------------------------
# Config gating tests
# ---------------------------------------------------------------------------


class TestSignalsGating:
    """Test that Signals features require the API key."""

    def test_config_defaults(self, monkeypatch):
        """Signals fields default to empty/default URL."""
        from mosaicx.config import MosaicxConfig

        monkeypatch.delenv("MOSAICX_SIGNALS_API_KEY", raising=False)
        monkeypatch.delenv("MOSAICX_SIGNALS_API_BASE", raising=False)
        cfg = MosaicxConfig(_env_file=None)
        assert cfg.signals_api_key == ""
        assert "supabase" in cfg.signals_api_base

    def test_config_env_override(self, monkeypatch):
        """MOSAICX_SIGNALS_API_KEY env var is picked up."""
        from mosaicx.config import MosaicxConfig

        monkeypatch.setenv("MOSAICX_SIGNALS_API_KEY", "sk_signals_test")
        cfg = MosaicxConfig(_env_file=None)
        assert cfg.signals_api_key == "sk_signals_test"

    def test_sdk_raises_without_key(self, monkeypatch):
        """signals_evaluate() raises ValueError without API key."""
        from mosaicx.sdk import signals_evaluate

        monkeypatch.setenv("MOSAICX_SIGNALS_API_KEY", "")

        # Clear cached config
        from mosaicx.config import get_config
        get_config.cache_clear()

        with pytest.raises(ValueError, match="Signals API key not configured"):
            signals_evaluate(ai_report_text="AI", reference_text="Ref", skip_deidentify=True)

        get_config.cache_clear()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestSignalsCLI:
    """Test CLI commands with mocked API calls."""

    def test_signals_hidden(self):
        """The signals group should be hidden from --help."""
        from click.testing import CliRunner

        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "signals" not in result.output

    def test_signals_health_no_key(self, monkeypatch):
        """signals health should error without API key."""
        from click.testing import CliRunner

        from mosaicx.cli import cli

        monkeypatch.setenv("MOSAICX_SIGNALS_API_KEY", "")
        from mosaicx.config import get_config
        get_config.cache_clear()

        runner = CliRunner()
        result = runner.invoke(cli, ["signals", "health"])
        assert result.exit_code != 0
        assert "API key" in result.output or "api key" in result.output.lower()

        get_config.cache_clear()

    def test_signals_health_success(self, monkeypatch):
        """signals health should display status on success."""
        from click.testing import CliRunner

        from mosaicx.cli import cli

        monkeypatch.setenv("MOSAICX_SIGNALS_API_KEY", "sk_signals_test")
        from mosaicx.config import get_config
        get_config.cache_clear()

        with patch("mosaicx.signals.SignalsClient") as MockClient:
            instance = MockClient.return_value
            instance.health.return_value = {
                "status": "ok",
                "service": "Signals API",
                "version": "v2.1",
            }

            runner = CliRunner()
            result = runner.invoke(cli, ["signals", "health"])

        assert result.exit_code == 0
        assert "online" in result.output or "ok" in result.output.lower()

        get_config.cache_clear()

    def test_signals_evaluate_missing_files(self, monkeypatch):
        """signals evaluate should error for non-existent files."""
        from click.testing import CliRunner

        from mosaicx.cli import cli

        monkeypatch.setenv("MOSAICX_SIGNALS_API_KEY", "sk_signals_test")
        from mosaicx.config import get_config
        get_config.cache_clear()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "signals", "evaluate",
            "--ai-report", "/nonexistent/ai.txt",
            "--reference", "/nonexistent/ref.txt",
        ])
        assert result.exit_code != 0

        get_config.cache_clear()

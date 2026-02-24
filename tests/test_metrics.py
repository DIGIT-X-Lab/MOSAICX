"""Tests for reward functions."""
import pytest

class TestExtractionReward:
    def test_penalizes_empty_findings(self):
        from mosaicx.evaluation.rewards import extraction_reward
        score = extraction_reward(findings=[], impression="Normal")
        assert score < 0.5

    def test_rewards_complete_extraction(self):
        from mosaicx.evaluation.rewards import extraction_reward
        score = extraction_reward(
            findings=[{"anatomy": "RUL", "observation": "nodule", "description": "5mm"}],
            impression="Pulmonary nodule, follow-up.",
        )
        assert score > 0.5

    def test_penalizes_findings_without_anatomy(self):
        from mosaicx.evaluation.rewards import extraction_reward
        low = extraction_reward(
            findings=[{"anatomy": "", "observation": "nodule", "description": "something"}],
            impression="Normal",
        )
        high = extraction_reward(
            findings=[{"anatomy": "RUL", "observation": "nodule", "description": "something"}],
            impression="Normal",
        )
        assert high > low

class TestPHILeakReward:
    def test_clean_text_scores_high(self):
        from mosaicx.evaluation.rewards import phi_leak_reward
        score = phi_leak_reward("The lungs are clear bilaterally.")
        assert score == 1.0

    def test_text_with_ssn_scores_zero(self):
        from mosaicx.evaluation.rewards import phi_leak_reward
        score = phi_leak_reward("Patient SSN 123-45-6789.")
        assert score == 0.0

    def test_text_with_phone_scores_zero(self):
        from mosaicx.evaluation.rewards import phi_leak_reward
        score = phi_leak_reward("Call (555) 123-4567.")
        assert score == 0.0


class TestHarmonyLMConfig:
    def test_normalize_local_api_base_localhost(self):
        from mosaicx.metrics import _normalize_local_api_base

        assert _normalize_local_api_base("http://localhost:8000/v1") == "http://127.0.0.1:8000/v1"

    def test_normalize_local_api_base_ipv6_loopback(self):
        from mosaicx.metrics import _normalize_local_api_base

        assert _normalize_local_api_base("http://[::1]:8000/v1") == "http://127.0.0.1:8000/v1"

    def test_normalize_local_api_base_keeps_remote_hosts(self):
        from mosaicx.metrics import _normalize_local_api_base

        assert _normalize_local_api_base("https://api.example.com/v1") == "https://api.example.com/v1"

    def test_normalize_model_for_local_openai_compatible_base_adds_prefix(self):
        from mosaicx.metrics import _normalize_model_for_api_base

        assert (
            _normalize_model_for_api_base(
                "mlx-community/gpt-oss-120b-4bit",
                "http://127.0.0.1:8000/v1",
            )
            == "openai/mlx-community/gpt-oss-120b-4bit"
        )

    def test_normalize_model_for_local_base_keeps_known_provider_prefix(self):
        from mosaicx.metrics import _normalize_model_for_api_base

        assert (
            _normalize_model_for_api_base(
                "openai/gpt-oss:120b",
                "http://127.0.0.1:8000/v1",
            )
            == "openai/gpt-oss:120b"
        )

    def test_normalize_model_for_remote_base_keeps_model_unchanged(self):
        from mosaicx.metrics import _normalize_model_for_api_base

        assert (
            _normalize_model_for_api_base(
                "mlx-community/gpt-oss-120b-4bit",
                "https://api.example.com/v1",
            )
            == "mlx-community/gpt-oss-120b-4bit"
        )

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

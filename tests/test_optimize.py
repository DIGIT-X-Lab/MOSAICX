"""Tests for the optimization workflow."""
import pytest
from pathlib import Path


class TestOptimizationConfig:
    def test_budget_presets(self):
        from mosaicx.evaluation.optimize import get_optimizer_config
        light = get_optimizer_config("light")
        medium = get_optimizer_config("medium")
        heavy = get_optimizer_config("heavy")
        assert light["max_iterations"] < medium["max_iterations"] < heavy["max_iterations"]

    def test_progressive_strategy(self):
        from mosaicx.evaluation.optimize import OPTIMIZATION_STRATEGY
        assert len(OPTIMIZATION_STRATEGY) == 3
        names = [s["name"] for s in OPTIMIZATION_STRATEGY]
        assert "BootstrapFewShot" in names
        assert "SIMBA" in names
        assert "GEPA" in names

    def test_budget_presets_have_required_keys(self):
        from mosaicx.evaluation.optimize import get_optimizer_config
        for budget in ("light", "medium", "heavy"):
            config = get_optimizer_config(budget)
            assert "max_iterations" in config
            assert "strategy" in config

    def test_list_pipelines_includes_query_and_verify(self):
        from mosaicx.evaluation.optimize import list_pipelines

        names = list_pipelines()
        assert "query" in names
        assert "verify" in names

    def test_get_pipeline_class_query_and_verify(self):
        from mosaicx.evaluation.optimize import get_pipeline_class

        query_cls = get_pipeline_class("query")
        verify_cls = get_pipeline_class("verify")
        assert query_cls.__name__ == "QueryGroundedResponder"
        assert verify_cls.__name__ == "VerifyClaimResponder"

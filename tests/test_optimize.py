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

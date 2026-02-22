# tests/test_optimize_simba.py
from __future__ import annotations


class TestSIMBABudget:
    def test_medium_budget_uses_simba(self):
        from mosaicx.evaluation.optimize import get_optimizer_config

        config = get_optimizer_config("medium")
        assert config["strategy"] == "SIMBA"

    def test_budget_presets_all_exist(self):
        from mosaicx.evaluation.optimize import get_optimizer_config

        for budget in ("light", "medium", "heavy"):
            config = get_optimizer_config(budget)
            assert "strategy" in config
            assert "max_iterations" in config

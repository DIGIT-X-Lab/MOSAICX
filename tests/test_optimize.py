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

    def test_get_strategy_config_supports_explicit_dspy_strategies(self):
        from mosaicx.evaluation.optimize import get_strategy_config

        for strategy in ("MIPROv2", "SIMBA", "GEPA"):
            config = get_strategy_config(strategy)
            assert config["strategy"] == strategy
            assert "max_iterations" in config
            assert "num_candidates" in config


def test_run_optimizer_sequence_writes_manifest_and_artifacts(tmp_path: Path, monkeypatch):
    from mosaicx.evaluation import optimize as opt

    calls: list[str] = []

    def _fake_run(
        *,
        module,
        trainset,
        valset,
        metric,
        strategy,
        save_path,
        config_override=None,
    ):
        Path(save_path).write_text("{}")
        calls.append(str(strategy))
        return object(), {
            "train_score": 0.9,
            "val_score": 0.8,
            "num_train": len(trainset),
            "num_val": len(valset or []),
            "strategy": str(strategy),
        }

    monkeypatch.setattr(opt, "run_optimization_with_strategy", _fake_run)
    manifest = opt.run_optimizer_sequence(
        module_factory=lambda: object(),
        trainset=[1, 2, 3],
        valset=[4],
        metric=lambda *_args, **_kwargs: 1.0,
        out_dir=tmp_path,
        strategies=("MIPROv2", "SIMBA", "GEPA"),
    )

    assert calls == ["MIPROv2", "SIMBA", "GEPA"]
    assert Path(manifest["manifest_path"]).exists()
    for run in manifest["runs"]:
        assert Path(run["artifact"]).exists()

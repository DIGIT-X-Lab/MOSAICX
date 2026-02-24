"""Optimization workflow — progressive DSPy optimizer strategy."""

from __future__ import annotations

import json
import logging
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mosaicx.runtime_env import import_dspy

logger = logging.getLogger(__name__)

OPTIMIZATION_STRATEGY: list[dict[str, Any]] = [
    {"name": "BootstrapFewShot", "cost": "~$0.50", "time": "~5 min", "min_examples": 10},
    {"name": "SIMBA", "cost": "~$3", "time": "~20 min", "min_examples": 10},
    {"name": "GEPA", "cost": "~$10", "time": "~45 min", "min_examples": 10},
]

_BUDGET_PRESETS: dict[str, dict[str, Any]] = {
    "light": {"max_iterations": 10, "strategy": "BootstrapFewShot", "num_candidates": 5},
    "medium": {"max_iterations": 50, "strategy": "SIMBA", "num_candidates": 10},
    "heavy": {"max_iterations": 150, "strategy": "GEPA", "num_candidates": 20,
              "reflection_lm": None, "candidate_selection_strategy": "pareto", "use_merge": True},
}

_STRATEGY_DEFAULTS: dict[str, dict[str, Any]] = {
    "BootstrapFewShot": {"max_iterations": 10, "num_candidates": 5},
    "MIPROv2": {"max_iterations": 50, "num_candidates": 10},
    "SIMBA": {"max_iterations": 50, "num_candidates": 10},
    "GEPA": {"max_iterations": 150, "num_candidates": 20,
             "reflection_lm": None, "candidate_selection_strategy": "pareto", "use_merge": True},
}


def get_optimizer_config(budget: str) -> dict[str, Any]:
    """Get optimizer configuration for a budget preset."""
    if budget not in _BUDGET_PRESETS:
        raise ValueError(f"Unknown budget: {budget}. Choose from: {list(_BUDGET_PRESETS)}")
    return dict(_BUDGET_PRESETS[budget])


def get_strategy_config(strategy: str) -> dict[str, Any]:
    """Get optimizer configuration for an explicit DSPy strategy name."""
    name = str(strategy).strip()
    if name not in _STRATEGY_DEFAULTS:
        raise ValueError(f"Unknown strategy: {name}. Choose from: {list(_STRATEGY_DEFAULTS)}")
    cfg = dict(_STRATEGY_DEFAULTS[name])
    cfg["strategy"] = name
    return cfg


def save_optimized(module: Any, path: Path) -> None:
    """Save an optimized DSPy module to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module.save(str(path))


def load_optimized(module_class: type, path: Path) -> Any:
    """Load an optimized DSPy module from disk."""
    module = module_class()
    module.load(str(path))
    return module


# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------

_PIPELINE_REGISTRY: dict[str, tuple[str, str]] = {
    "radiology":    ("mosaicx.pipelines.radiology",    "RadiologyReportStructurer"),
    "pathology":    ("mosaicx.pipelines.pathology",    "PathologyReportStructurer"),
    "extract":      ("mosaicx.pipelines.extraction",   "DocumentExtractor"),
    "summarize":    ("mosaicx.pipelines.summarizer",   "ReportSummarizer"),
    "deidentify":   ("mosaicx.pipelines.deidentifier", "Deidentifier"),
    "schema":       ("mosaicx.pipelines.schema_gen",   "SchemaGenerator"),
    "query":        ("mosaicx.pipelines.query_qa",     "QueryGroundedResponder"),
    "verify":       ("mosaicx.pipelines.verify_claim", "VerifyClaimResponder"),
}


def list_pipelines() -> list[str]:
    """Return sorted list of available pipeline names."""
    return sorted(_PIPELINE_REGISTRY)


def get_pipeline_class(pipeline: str) -> type:
    """Lazily import and return the DSPy Module class for *pipeline*.

    Raises
    ------
    ValueError
        If *pipeline* is not recognised.
    ImportError
        If the pipeline module cannot be imported.
    """
    if pipeline not in _PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'. "
            f"Available: {list_pipelines()}"
        )

    module_path, class_name = _PIPELINE_REGISTRY[pipeline]

    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls


# ---------------------------------------------------------------------------
# Optimization execution
# ---------------------------------------------------------------------------

def _build_optimizer(strategy: str, config: dict[str, Any], metric: Callable) -> Any:
    """Instantiate a DSPy optimizer by strategy name.

    Falls back to BootstrapFewShot if the requested optimizer is
    unavailable.
    """
    dspy = import_dspy()

    max_rounds = config.get("max_iterations", 10)

    def _instantiate(ctor: Any, **kwargs: Any) -> Any:
        """Instantiate optimizer while filtering kwargs to supported params."""
        sig = inspect.signature(ctor)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_var_kwargs:
            filtered = kwargs
        else:
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return ctor(**filtered)

    if strategy == "BootstrapFewShot":
        return dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=config.get("num_candidates", 5),
            max_rounds=max_rounds,
        )

    if strategy == "SIMBA":
        simba_cls = getattr(dspy, "SIMBA", None)
        if simba_cls is None:
            logger.warning("SIMBA unavailable, falling back to BootstrapFewShot")
            return dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=config.get("num_candidates", 5),
                max_rounds=max_rounds,
            )
        try:
            simba_steps = int(
                config.get(
                    "max_steps",
                    max(2, min(int(max_rounds // 10), 8)),
                )
            )
            return _instantiate(
                simba_cls,
                metric=metric,
                num_candidates=config.get("num_candidates", 10),
                bsize=min(int(config.get("num_candidates", 10)), 32),
                max_steps=max(1, simba_steps),
                max_demos=4,
                num_threads=1,
            )
        except (AttributeError, TypeError, ValueError):
            logger.warning("SIMBA init failed, falling back to BootstrapFewShot")
            return dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=config.get("num_candidates", 5),
                max_rounds=max_rounds,
            )

    if strategy == "MIPROv2":
        try:
            # Newer DSPy raises when auto != None and num_candidates is set.
            return _instantiate(
                dspy.MIPROv2,
                metric=metric,
                auto=None,
                num_candidates=config.get("num_candidates", 10),
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
                num_threads=1,
            )
        except (AttributeError, TypeError, ValueError):
            try:
                # Compatibility fallback for older/newer constructor signatures.
                return _instantiate(
                    dspy.MIPROv2,
                    metric=metric,
                    auto="light",
                    max_bootstrapped_demos=4,
                    max_labeled_demos=4,
                    num_threads=1,
                )
            except (AttributeError, TypeError, ValueError):
                logger.warning("MIPROv2 unavailable, falling back to BootstrapFewShot")
                return dspy.BootstrapFewShot(
                    metric=metric,
                    max_bootstrapped_demos=config.get("num_candidates", 5),
                    max_rounds=max_rounds,
                )

    if strategy == "GEPA":
        # GEPA may not be available in all DSPy versions
        gepa_cls = getattr(dspy, "GEPA", None) or getattr(dspy, "BootstrapFewShotWithRandomSearch", None)
        if gepa_cls is None:
            logger.warning("GEPA unavailable, falling back to MIPROv2")
            return _build_optimizer("MIPROv2", config, metric)
        try:
            return _instantiate(
                gepa_cls,
                metric=metric,
                auto=None,
                max_bootstrapped_demos=config.get("num_candidates", 20),
                num_candidate_programs=config.get("num_candidates", 20),
                num_threads=1,
                candidate_selection_strategy=config.get("candidate_selection_strategy", "pareto"),
                reflection_lm=config.get("reflection_lm"),
                use_merge=bool(config.get("use_merge", True)),
            )
        except (TypeError, ValueError):
            logger.warning("GEPA init failed, falling back to BootstrapFewShot")
            return dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=config.get("num_candidates", 5),
                max_rounds=max_rounds,
            )

    raise ValueError(f"Unknown optimizer strategy: {strategy}")


def _compile_optimizer(
    *,
    optimizer: Any,
    module: Any,
    trainset: list,
    valset: list | None,
    strategy: str,
    config: dict[str, Any],
) -> Any:
    """Compile optimizer with signature-aware kwargs across DSPy versions."""
    compile_sig = inspect.signature(optimizer.compile)
    kwargs: dict[str, Any] = {}

    if "trainset" in compile_sig.parameters:
        kwargs["trainset"] = trainset
    if valset is not None and "valset" in compile_sig.parameters:
        kwargs["valset"] = valset
    if "seed" in compile_sig.parameters:
        kwargs["seed"] = 9

    if strategy == "MIPROv2":
        if "num_trials" in compile_sig.parameters:
            kwargs["num_trials"] = max(
                int(config.get("num_trials", config.get("max_iterations", 10))),
                1,
            )
        if "max_bootstrapped_demos" in compile_sig.parameters:
            kwargs["max_bootstrapped_demos"] = 4
        if "max_labeled_demos" in compile_sig.parameters:
            kwargs["max_labeled_demos"] = 4
        if "minibatch_size" in compile_sig.parameters:
            val_size = len(valset) if isinstance(valset, list) else len(trainset)
            kwargs["minibatch_size"] = max(1, min(4, int(val_size)))
        if "minibatch_full_eval_steps" in compile_sig.parameters:
            kwargs["minibatch_full_eval_steps"] = 1

    return optimizer.compile(module, **kwargs)


def run_optimization(
    module: Any,
    trainset: list,
    valset: list | None,
    metric: Callable,
    budget: str = "light",
    save_path: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run DSPy optimization on *module* using *trainset*.

    Parameters
    ----------
    module:
        An instantiated DSPy Module to optimize.
    trainset:
        Training examples (``dspy.Example`` objects).
    valset:
        Validation examples.  If ``None``, *trainset* is split 80/20.
    metric:
        Metric function ``(example, prediction, trace) -> float``.
    budget:
        Budget preset name (``"light"``, ``"medium"``, ``"heavy"``).
    save_path:
        If provided, save the optimized module here.

    Returns
    -------
    tuple[optimized_module, results_dict]
        ``results_dict`` has keys ``train_score``, ``val_score``,
        ``num_train``, ``num_val``, ``strategy``.
    """
    dspy = import_dspy()

    config = get_optimizer_config(budget)
    strategy = config["strategy"]

    # Auto-split if no valset
    if valset is None and len(trainset) >= 5:
        split = int(len(trainset) * 0.8)
        valset = trainset[split:]
        trainset = trainset[:split]
    elif valset is None:
        valset = trainset  # Tiny dataset — reuse train as val

    # Build and run optimizer
    optimizer = _build_optimizer(strategy, config, metric)
    optimized = _compile_optimizer(
        optimizer=optimizer,
        module=module,
        trainset=trainset,
        valset=valset,
        strategy=str(strategy),
        config=config,
    )

    # Evaluate on train + val sets
    evaluator = dspy.Evaluate(
        devset=trainset,
        metric=metric,
        num_threads=1,
        display_progress=True,
    )
    train_score = evaluator(optimized)

    evaluator_val = dspy.Evaluate(
        devset=valset,
        metric=metric,
        num_threads=1,
        display_progress=True,
    )
    val_score = evaluator_val(optimized)

    # Save if requested
    if save_path:
        save_optimized(optimized, Path(save_path))

    results = {
        "train_score": train_score,
        "val_score": val_score,
        "num_train": len(trainset),
        "num_val": len(valset),
        "strategy": strategy,
    }

    return optimized, results


def run_optimization_with_strategy(
    module: Any,
    trainset: list,
    valset: list | None,
    metric: Callable,
    strategy: str,
    save_path: Path | str | None = None,
    config_override: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run optimization with an explicit DSPy strategy (e.g. MIPROv2/SIMBA/GEPA)."""
    dspy = import_dspy()

    config = get_strategy_config(strategy)
    if config_override:
        config.update({k: v for k, v in config_override.items() if v is not None})

    if valset is None and len(trainset) >= 5:
        split = int(len(trainset) * 0.8)
        valset = trainset[split:]
        trainset = trainset[:split]
    elif valset is None:
        valset = trainset

    optimizer = _build_optimizer(str(strategy), config, metric)
    optimized = _compile_optimizer(
        optimizer=optimizer,
        module=module,
        trainset=trainset,
        valset=valset,
        strategy=str(strategy),
        config=config,
    )

    evaluator = dspy.Evaluate(
        devset=trainset,
        metric=metric,
        num_threads=1,
        display_progress=True,
    )
    train_score = evaluator(optimized)

    evaluator_val = dspy.Evaluate(
        devset=valset,
        metric=metric,
        num_threads=1,
        display_progress=True,
    )
    val_score = evaluator_val(optimized)

    if save_path:
        save_optimized(optimized, Path(save_path))

    results = {
        "train_score": train_score,
        "val_score": val_score,
        "num_train": len(trainset),
        "num_val": len(valset),
        "strategy": str(strategy),
    }
    return optimized, results


def run_optimizer_sequence(
    *,
    module_factory: Callable[[], Any],
    trainset: list,
    valset: list | None,
    metric: Callable,
    out_dir: Path | str,
    strategies: tuple[str, ...] = ("MIPROv2", "SIMBA", "GEPA"),
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run and persist a full optimizer sequence, returning manifest metadata."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    baseline: dict[str, Any] | None = None
    baseline_artifact: str | None = None
    try:
        dspy = import_dspy()
        baseline_module = module_factory()
        evaluator_train = dspy.Evaluate(
            devset=trainset,
            metric=metric,
            num_threads=1,
            display_progress=True,
        )
        baseline_train = float(evaluator_train(baseline_module))
        effective_valset = valset if valset is not None else trainset
        evaluator_val = dspy.Evaluate(
            devset=effective_valset,
            metric=metric,
            num_threads=1,
            display_progress=True,
        )
        baseline_val = float(evaluator_val(baseline_module))
        baseline = {
            "train_score": baseline_train,
            "val_score": baseline_val,
            "num_train": len(trainset),
            "num_val": len(effective_valset),
        }
        baseline_path = out_path / "baseline_metrics.json"
        baseline_path.write_text(json.dumps(baseline, indent=2))
        baseline_artifact = str(baseline_path)
    except Exception as exc:
        logger.warning("Baseline evaluation failed; continuing optimizer sequence: %s", exc)

    runs: list[dict[str, Any]] = []
    for strategy in strategies:
        save_path = out_path / f"{str(strategy).lower()}_optimized.json"
        module = module_factory()
        _, result = run_optimization_with_strategy(
            module=module,
            trainset=trainset,
            valset=valset,
            metric=metric,
            strategy=str(strategy),
            save_path=save_path,
            config_override=(strategy_overrides or {}).get(str(strategy)),
        )
        runs.append(
            {
                "strategy": str(strategy),
                "train_score": float(result["train_score"]),
                "val_score": float(result["val_score"]),
                "num_train": int(result["num_train"]),
                "num_val": int(result["num_val"]),
                "artifact": str(save_path),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategies": [str(s) for s in strategies],
        "strategy_overrides": strategy_overrides or {},
        "baseline": baseline,
        "baseline_artifact": baseline_artifact,
        "runs": runs,
    }
    manifest_path = out_path / "optimizer_sequence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["manifest_path"] = str(manifest_path)
    return manifest

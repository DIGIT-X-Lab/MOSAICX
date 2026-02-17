"""Optimization workflow — progressive GEPA strategy."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

OPTIMIZATION_STRATEGY: list[dict[str, Any]] = [
    {"name": "BootstrapFewShot", "cost": "~$0.50", "time": "~5 min", "min_examples": 10},
    {"name": "MIPROv2", "cost": "~$3", "time": "~20 min", "min_examples": 10},
    {"name": "GEPA", "cost": "~$10", "time": "~45 min", "min_examples": 10},
]

_BUDGET_PRESETS: dict[str, dict[str, Any]] = {
    "light": {"max_iterations": 10, "strategy": "BootstrapFewShot", "num_candidates": 5},
    "medium": {"max_iterations": 50, "strategy": "MIPROv2", "num_candidates": 10},
    "heavy": {"max_iterations": 150, "strategy": "GEPA", "num_candidates": 20,
              "reflection_lm": None, "candidate_selection_strategy": "pareto", "use_merge": True},
}


def get_optimizer_config(budget: str) -> dict[str, Any]:
    """Get optimizer configuration for a budget preset."""
    if budget not in _BUDGET_PRESETS:
        raise ValueError(f"Unknown budget: {budget}. Choose from: {list(_BUDGET_PRESETS)}")
    return dict(_BUDGET_PRESETS[budget])


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
    import dspy

    max_rounds = config.get("max_iterations", 10)

    if strategy == "BootstrapFewShot":
        return dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=config.get("num_candidates", 5),
            max_rounds=max_rounds,
        )

    if strategy == "MIPROv2":
        try:
            return dspy.MIPROv2(
                metric=metric,
                num_candidates=config.get("num_candidates", 10),
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
            )
        except (AttributeError, TypeError):
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
            return gepa_cls(
                metric=metric,
                max_bootstrapped_demos=config.get("num_candidates", 20),
                num_threads=1,
            )
        except TypeError:
            logger.warning("GEPA init failed, falling back to BootstrapFewShot")
            return dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=config.get("num_candidates", 5),
                max_rounds=max_rounds,
            )

    raise ValueError(f"Unknown optimizer strategy: {strategy}")


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
    import dspy

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
    optimized = optimizer.compile(module, trainset=trainset)

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

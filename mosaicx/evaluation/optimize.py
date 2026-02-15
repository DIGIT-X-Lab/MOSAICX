"""Optimization workflow — progressive GEPA strategy."""

from __future__ import annotations
from pathlib import Path
from typing import Any

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

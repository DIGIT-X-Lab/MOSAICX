"""Quality, metrics & optimization.

Submodules:
    dataset   — JSONL loader for DSPy optimization/evaluation
    metrics   — per-pipeline DSPy metric functions
    optimize  — budget presets, optimizer execution, save/load
    rewards   — scalar reward functions (extraction_reward, phi_leak_reward)
    completeness — field coverage & information density
"""

from .dataset import PIPELINE_INPUT_FIELDS, load_jsonl
from .metrics import get_metric, list_metrics
from .grounding import (
    complete_and_grounded_metric,
    numeric_exactness_metric,
    query_metric_components,
    query_grounded_numeric_metric,
    verify_metric_components,
    verify_grounded_metric,
)
from .quality_gates import (
    DEFAULT_QUERY_GATES,
    DEFAULT_VERIFY_GATES,
    evaluate_quality_gates,
    evaluate_query_quality_gates,
    evaluate_verify_quality_gates,
)
from .optimize import (
    OPTIMIZATION_STRATEGY,
    get_optimizer_config,
    get_pipeline_class,
    list_pipelines,
    load_optimized,
    run_optimization,
    save_optimized,
)

__all__ = [
    "OPTIMIZATION_STRATEGY",
    "PIPELINE_INPUT_FIELDS",
    "complete_and_grounded_metric",
    "DEFAULT_QUERY_GATES",
    "DEFAULT_VERIFY_GATES",
    "evaluate_quality_gates",
    "evaluate_query_quality_gates",
    "evaluate_verify_quality_gates",
    "get_metric",
    "get_optimizer_config",
    "get_pipeline_class",
    "list_metrics",
    "list_pipelines",
    "load_jsonl",
    "load_optimized",
    "numeric_exactness_metric",
    "query_metric_components",
    "query_grounded_numeric_metric",
    "run_optimization",
    "save_optimized",
    "verify_metric_components",
    "verify_grounded_metric",
]

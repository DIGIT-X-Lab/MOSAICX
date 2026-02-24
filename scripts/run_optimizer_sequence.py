#!/usr/bin/env python3
"""Run MIPROv2 -> SIMBA -> GEPA optimization and persist artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, help="Pipeline name (e.g. query, verify, radiology).")
    parser.add_argument("--trainset", required=True, type=Path, help="Training JSONL path.")
    parser.add_argument("--valset", default=None, type=Path, help="Optional validation JSONL path.")
    parser.add_argument(
        "--strategies",
        default="MIPROv2,SIMBA,GEPA",
        help="Comma-separated optimizer strategies to run in order.",
    )
    parser.add_argument(
        "--profile",
        choices=("standard", "quick"),
        default="quick",
        help="Optimizer profile: quick for bounded local runs, standard for heavier search.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where optimizer artifacts and manifest are written.",
    )
    args = parser.parse_args()

    from mosaicx.evaluation.dataset import load_jsonl
    from mosaicx.evaluation.metrics import get_metric
    from mosaicx.evaluation.optimize import get_pipeline_class, run_optimizer_sequence
    from mosaicx.runtime_env import import_dspy
    from mosaicx.sdk import _ensure_configured

    dspy = import_dspy()
    if getattr(dspy.settings, "lm", None) is None:
        _ensure_configured()

    pipeline_cls = get_pipeline_class(args.pipeline)
    metric = get_metric(args.pipeline)
    train_examples = load_jsonl(args.trainset, args.pipeline)
    val_examples = load_jsonl(args.valset, args.pipeline) if args.valset else None
    strategies = tuple(s.strip() for s in str(args.strategies).split(",") if s.strip())
    if not strategies:
        raise ValueError("No strategies provided.")

    strategy_overrides: dict[str, dict[str, int]] = {}
    if args.profile == "quick":
        strategy_overrides = {
            "MIPROv2": {"max_iterations": 2, "num_trials": 2, "num_candidates": 2},
            "SIMBA": {"max_iterations": 4, "num_candidates": 2, "max_steps": 2},
            "GEPA": {"max_iterations": 4, "num_candidates": 2},
        }

    manifest = run_optimizer_sequence(
        module_factory=pipeline_cls,
        trainset=train_examples,
        valset=val_examples,
        metric=metric,
        out_dir=args.out_dir,
        strategies=strategies,
        strategy_overrides=strategy_overrides,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# mosaicx/pipelines/scaffold.py
"""Scaffold a new extraction pipeline from the built-in template.

Usage::

    from mosaicx.pipelines.scaffold import scaffold_pipeline
    path = scaffold_pipeline("cardiology", "Cardiology report structurer")
"""

from __future__ import annotations

import re
from pathlib import Path

from mosaicx.pipelines._template import PIPELINE_TEMPLATE


def _to_snake_case(name: str) -> str:
    """Convert an arbitrary string to snake_case.

    Handles PascalCase, camelCase, kebab-case, and space-separated words.
    """
    # Replace hyphens and spaces with underscores
    s = name.replace("-", "_").replace(" ", "_")
    # Insert underscores before uppercase letters (for PascalCase / camelCase)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def _to_pascal_case(snake: str) -> str:
    """Convert a snake_case string to PascalCase."""
    return "".join(word.capitalize() for word in snake.split("_"))


def scaffold_pipeline(
    name: str,
    description: str = "",
) -> Path:
    """Generate a new pipeline module from the built-in template.

    Parameters
    ----------
    name:
        Pipeline name.  Will be normalised to snake_case for the file name
        and PascalCase for the class name.
    description:
        One-line human-readable description.  Falls back to a generic
        description if not provided.

    Returns
    -------
    Path
        Absolute path of the created pipeline file.

    Raises
    ------
    FileExistsError
        If the target file already exists (we never overwrite).
    ValueError
        If the name is empty or invalid after normalisation.
    """
    snake = _to_snake_case(name)
    if not snake:
        raise ValueError(f"Invalid pipeline name: {name!r}")

    pascal = _to_pascal_case(snake)
    class_name = f"{pascal}ReportStructurer"
    # Short class name for the Signature (e.g. "Cardiology" from "CardiologyReportStructurer")
    class_name_short = pascal

    if not description:
        description = f"Single-step {snake.replace('_', ' ')} document extraction pipeline"

    pipelines_dir = Path(__file__).resolve().parent
    target = pipelines_dir / f"{snake}.py"

    if target.exists():
        raise FileExistsError(
            f"Pipeline file already exists: {target}\n"
            "Remove it first if you want to regenerate."
        )

    source = PIPELINE_TEMPLATE.format(
        name=snake,
        class_name=class_name,
        class_name_short=class_name_short,
        description=description,
    )

    target.write_text(source, encoding="utf-8")

    return target


# ---------------------------------------------------------------------------
# Manual wiring checklist
# ---------------------------------------------------------------------------

WIRING_CHECKLIST: list[str] = [
    "Add to mosaicx/pipelines/modes.py _MODE_MODULES dict:",
    '    "{name}": "mosaicx.pipelines.{name}",',
    "",
    "Add to mosaicx/pipelines/modes.py _trigger_lazy_load() _LAZY_CLASS_NAMES dict:",
    '    "{name}": "{class_name}ReportStructurer",',
    "",
    "Add to mosaicx/evaluation/dataset.py PIPELINE_INPUT_FIELDS dict:",
    '    "{name}": ["document_text"],',
    "",
    "Add a metric to mosaicx/evaluation/metrics.py _METRIC_REGISTRY dict:",
    '    "{name}": <your_metric_function>,',
    "",
    "Add to mosaicx/evaluation/optimize.py _PIPELINE_REGISTRY dict:",
    '    "{name}": ("mosaicx.pipelines.{name}", "{class_name}ReportStructurer"),',
    "",
    "Wire CLI: add `import mosaicx.pipelines.{name}` alongside other",
    "    pipeline imports in mosaicx/cli.py (extract and batch commands).",
]

"""JSONL dataset loader for DSPy optimization and evaluation.

Reads line-delimited JSON files and converts each record into a
``dspy.Example`` with the correct ``.with_inputs()`` partitioning based
on which pipeline the dataset targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy

# ---------------------------------------------------------------------------
# Pipeline → input field registry
# ---------------------------------------------------------------------------
# Keys listed here become *inputs* passed to the DSPy module.  Every
# other key in the JSONL record is treated as a gold-standard *label*
# for the optimizer / evaluator to score against.

PIPELINE_INPUT_FIELDS: dict[str, list[str]] = {
    "radiology": ["report_text", "report_header"],
    "pathology": ["report_text", "report_header"],
    "extract": ["document_text"],
    "summarize": ["reports", "patient_id"],
    "deidentify": ["document_text", "mode"],
    "schema": ["description", "example_text", "document_text"],
}


def load_jsonl(
    path: str | Path,
    pipeline: str,
) -> list[dspy.Example]:
    """Load a JSONL file into a list of ``dspy.Example`` objects.

    Parameters
    ----------
    path:
        Path to a ``.jsonl`` file.  Each line must be a valid JSON object.
    pipeline:
        Pipeline name (must be a key in :data:`PIPELINE_INPUT_FIELDS`).

    Returns
    -------
    list[dspy.Example]
        Examples with ``.with_inputs()`` already called so DSPy knows
        which fields are inputs and which are labels.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *pipeline* is not recognised or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if pipeline not in PIPELINE_INPUT_FIELDS:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'. "
            f"Available: {sorted(PIPELINE_INPUT_FIELDS)}"
        )

    input_fields = PIPELINE_INPUT_FIELDS[pipeline]
    examples: list[dspy.Example] = []

    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
            example = dspy.Example(**record).with_inputs(*input_fields)
            examples.append(example)

    if not examples:
        raise ValueError(f"No examples found in {path}")

    return examples

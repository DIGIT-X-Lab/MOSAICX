# mosaicx/evaluation/rewards.py
"""Reward functions for structured extraction quality and safety.

Provides scalar reward signals suitable for RLHF / GRPO optimization
loops or standalone evaluation.

Key functions:
    - extraction_reward: Scores the quality of a structured extraction
      based on finding completeness, anatomical grounding, and
      impression quality.
    - phi_leak_reward: Binary safety reward that detects accidental PHI
      leakage in generated text using the regex patterns from the
      deidentifier pipeline.
"""

from __future__ import annotations

import re
from typing import Any

from mosaicx.pipelines.deidentifier import PHI_PATTERNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MEASUREMENT_PATTERN = re.compile(
    r"\d+\s*(mm|cm|m|cc|ml|mg|g|kg|in|ft)", re.IGNORECASE
)


def _has_measurement(text: str) -> bool:
    """Return True if *text* contains a measurement-like substring."""
    return bool(_MEASUREMENT_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Extraction reward
# ---------------------------------------------------------------------------


def extraction_reward(
    findings: list[dict[str, Any]],
    impression: str,
) -> float:
    """Score the quality of a structured extraction.

    The reward is computed as a sum of components (capped at 1.0):

    * +0.3 if *findings* is non-empty.
    * +0.1 per finding with a non-empty ``"anatomy"`` key (max 0.4).
    * +0.2 if *impression* is non-empty and longer than 10 characters.
    * +0.1 bonus per finding containing a measurement-like token.

    Parameters
    ----------
    findings:
        List of finding dicts.  Each dict may contain keys like
        ``"anatomy"``, ``"observation"``, ``"description"``.
    impression:
        The impression / conclusion string.

    Returns
    -------
    float
        Reward score in [0.0, 1.0].
    """
    score = 0.0

    # Non-empty findings
    if findings:
        score += 0.3

    # Anatomy grounding
    anatomy_bonus = 0.0
    for finding in findings:
        anatomy = finding.get("anatomy", "")
        if anatomy and anatomy.strip():
            anatomy_bonus += 0.1
    score += min(anatomy_bonus, 0.4)

    # Impression quality
    if impression and len(impression.strip()) > 10:
        score += 0.2

    # Measurement bonus
    measurement_bonus = 0.0
    for finding in findings:
        description = finding.get("description", "")
        observation = finding.get("observation", "")
        text = f"{description} {observation}"
        if _has_measurement(text):
            measurement_bonus += 0.1
    score += measurement_bonus

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# PHI leak reward
# ---------------------------------------------------------------------------


def phi_leak_reward(text: str) -> float:
    """Return 1.0 if *text* contains no detectable PHI, else 0.0.

    Uses the compiled regex patterns from
    :data:`mosaicx.pipelines.deidentifier.PHI_PATTERNS` to scan for
    format-based PHI (SSNs, phone numbers, MRNs, emails, US-format
    dates).

    Parameters
    ----------
    text:
        The text to scan for PHI leakage.

    Returns
    -------
    float
        1.0 if clean, 0.0 if any PHI pattern matches.
    """
    for pattern in PHI_PATTERNS:
        if pattern.search(text):
            return 0.0
    return 1.0

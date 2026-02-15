"""Radiology scoring system enums and description look-ups.

Provides standardised string enums for common radiological classification
systems so that structured reports can reference categories in a type-safe
manner.
"""

from __future__ import annotations

from enum import Enum


# ---------------------------------------------------------------------------
# Lung-RADS
# ---------------------------------------------------------------------------

class LungRADS(str, Enum):
    """ACR Lung-RADS assessment categories for lung CT screening."""
    CATEGORY_1 = "1"
    CATEGORY_2 = "2"
    CATEGORY_3 = "3"
    CATEGORY_4A = "4A"
    CATEGORY_4B = "4B"
    CATEGORY_4X = "4X"


# ---------------------------------------------------------------------------
# BI-RADS
# ---------------------------------------------------------------------------

class BIRADS(str, Enum):
    """ACR BI-RADS assessment categories for breast imaging."""
    CATEGORY_0 = "0"
    CATEGORY_1 = "1"
    CATEGORY_2 = "2"
    CATEGORY_3 = "3"
    CATEGORY_4 = "4"
    CATEGORY_5 = "5"
    CATEGORY_6 = "6"


# ---------------------------------------------------------------------------
# TI-RADS
# ---------------------------------------------------------------------------

class TIRADS(str, Enum):
    """ACR TI-RADS assessment categories for thyroid nodules."""
    TR1 = "TR1"
    TR2 = "TR2"
    TR3 = "TR3"
    TR4 = "TR4"
    TR5 = "TR5"


# ---------------------------------------------------------------------------
# Deauville (5-point scale for PET/CT)
# ---------------------------------------------------------------------------

class Deauville(str, Enum):
    """Deauville 5-point scale for FDG PET/CT in lymphoma."""
    SCORE_1 = "1"
    SCORE_2 = "2"
    SCORE_3 = "3"
    SCORE_4 = "4"
    SCORE_5 = "5"


# ---------------------------------------------------------------------------
# Clinical descriptions look-up
# ---------------------------------------------------------------------------

_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "Lung-RADS": {
        "1": "Negative: no lung nodules or definitely benign nodules",
        "2": "Benign appearance: nodules with a very low likelihood of malignancy",
        "3": "Probably benign: short-term follow-up suggested",
        "4A": "Suspicious: additional diagnostic testing recommended",
        "4B": "Very suspicious: additional diagnostic testing and/or tissue sampling recommended",
        "4X": "Category 3 or 4 with additional suspicious features",
    },
    "BI-RADS": {
        "0": "Incomplete: additional imaging needed",
        "1": "Negative: no significant findings",
        "2": "Benign: non-cancerous findings",
        "3": "Probably benign: short-interval follow-up recommended",
        "4": "Suspicious: biopsy should be considered",
        "5": "Highly suggestive of malignancy: biopsy strongly recommended",
        "6": "Known biopsy-proven malignancy",
    },
    "TI-RADS": {
        "TR1": "Benign: no FNA recommended",
        "TR2": "Not suspicious: no FNA recommended",
        "TR3": "Mildly suspicious: FNA if >= 2.5 cm; follow-up if >= 1.5 cm",
        "TR4": "Moderately suspicious: FNA if >= 1.5 cm; follow-up if >= 1.0 cm",
        "TR5": "Highly suspicious: FNA if >= 1.0 cm; follow-up if >= 0.5 cm",
    },
    "Deauville": {
        "1": "No uptake above background",
        "2": "Uptake <= mediastinum",
        "3": "Uptake > mediastinum but <= liver",
        "4": "Uptake moderately increased compared to liver",
        "5": "Uptake markedly increased compared to liver and/or new sites of disease",
    },
}


def get_scoring_description(system_name: str, value: str) -> str:
    """Return a brief clinical description for a scoring category.

    Parameters
    ----------
    system_name:
        Name of the scoring system (e.g. ``"Lung-RADS"``, ``"BI-RADS"``).
    value:
        Category value within that system (e.g. ``"4B"``, ``"3"``).

    Returns
    -------
    str
        A human-readable description, or ``""`` if the system or value is
        not recognised.
    """
    return _DESCRIPTIONS.get(system_name, {}).get(value, "")

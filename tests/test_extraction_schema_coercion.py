from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from mosaicx.pipelines.extraction import (
    _coerce_payload_to_schema,
    _recover_schema_instance_from_raw,
)


class LabelAssessment(BaseModel):
    value: Literal[-1.0, 0.0, 1.0]
    supporting_text: str | None = None
    confidence: float | None = None


class MimicCXRNested(BaseModel):
    pleural_effusion: LabelAssessment | None = None
    pneumonia: LabelAssessment | None = None
    no_finding: LabelAssessment | None = None


class MimicCXRMixed(BaseModel):
    edema: Literal[-1.0, 0.0, 1.0] | None = None
    cardiomegaly: LabelAssessment | None = None
    pleural_effusion: LabelAssessment | None = None
    pneumothorax: LabelAssessment | None = None


def test_coerce_payload_wraps_scalar_into_nested_model():
    payload = {
        "pleural_effusion": "extensive right pleural effusion",
        "pneumonia": True,
        "no_finding": False,
    }
    coerced = _coerce_payload_to_schema(payload, MimicCXRNested)
    parsed = MimicCXRNested.model_validate(coerced)

    assert parsed.pleural_effusion is not None
    assert parsed.pleural_effusion.value == 1.0
    assert parsed.pleural_effusion.supporting_text == "extensive right pleural effusion"
    assert parsed.pneumonia is not None and parsed.pneumonia.value == 1.0
    assert parsed.no_finding is not None and parsed.no_finding.value == 0.0


def test_coerce_payload_unwraps_nested_value_for_literal_field():
    payload = {
        "cardiomegaly": {
            "value": 1.0,
            "supporting_text": "mild cardiomegaly",
            "confidence": None,
        },
        "edema": {
            "value": 1.0,
            "supporting_text": "moderate pulmonary edema",
            "confidence": None,
        },
        "pleural_effusion": {
            "value": 1.0,
            "supporting_text": "small pleural effusions",
            "confidence": None,
        },
        "pneumothorax": {
            "value": 0.0,
            "supporting_text": "No pneumothorax.",
            "confidence": None,
        },
    }

    coerced = _coerce_payload_to_schema(payload, MimicCXRMixed)
    parsed = MimicCXRMixed.model_validate(coerced)

    assert parsed.edema == 1.0
    assert parsed.cardiomegaly is not None and parsed.cardiomegaly.value == 1.0
    assert parsed.pleural_effusion is not None and parsed.pleural_effusion.value == 1.0
    assert parsed.pneumothorax is not None and parsed.pneumothorax.value == 0.0


def test_recover_schema_instance_from_prose_wrapped_json():
    raw = """
    We need to produce JSON object with keys for each allowed top-level key.
    ```json
    {
      "pleural_effusion": "extensive right pleural effusion",
      "pneumonia": true,
      "no_finding": false
    }
    ```
    """

    parsed = _recover_schema_instance_from_raw(raw, MimicCXRNested)
    assert isinstance(parsed, MimicCXRNested)
    assert parsed.pleural_effusion is not None
    assert parsed.pleural_effusion.value == 1.0
    assert parsed.pneumonia is not None and parsed.pneumonia.value == 1.0
    assert parsed.no_finding is not None and parsed.no_finding.value == 0.0


def test_recover_schema_instance_raises_for_non_json_text():
    with pytest.raises(ValueError):
        _recover_schema_instance_from_raw(
            "I cannot comply with JSON output for this task.",
            MimicCXRNested,
        )

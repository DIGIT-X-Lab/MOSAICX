"""Tests for evaluation dataset pipeline input-field mapping."""

from __future__ import annotations

import json


def test_pipeline_input_fields_include_query_and_verify():
    from mosaicx.evaluation.dataset import PIPELINE_INPUT_FIELDS

    assert PIPELINE_INPUT_FIELDS["query"] == ["question", "source_text"]
    assert PIPELINE_INPUT_FIELDS["verify"] == ["claim", "source_text"]


def test_load_jsonl_for_query_pipeline(tmp_path):
    from mosaicx.evaluation.dataset import load_jsonl

    data_path = tmp_path / "query.jsonl"
    row = {
        "question": "What modality was used?",
        "source_text": "Study Date: 2025-09-10\nModality: CT",
        "response": "CT",
        "expected_numeric": None,
    }
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    examples = load_jsonl(data_path, "query")
    assert len(examples) == 1

    inputs = dict(examples[0].inputs())
    assert set(inputs) == {"question", "source_text"}


def test_load_jsonl_for_verify_pipeline(tmp_path):
    from mosaicx.evaluation.dataset import load_jsonl

    data_path = tmp_path / "verify.jsonl"
    row = {
        "claim": "patient BP is 128/82",
        "source_text": "Blood pressure: 128/82",
        "verdict": "verified",
    }
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    examples = load_jsonl(data_path, "verify")
    assert len(examples) == 1

    inputs = dict(examples[0].inputs())
    assert set(inputs) == {"claim", "source_text"}

# tests/test_template_compiler.py
"""Tests for the YAML template compiler."""

import pytest
import yaml
from pathlib import Path
from pydantic import BaseModel


SAMPLE_TEMPLATE_YAML = """\
name: ChestCTReport
description: "Structured chest CT report"

sections:
  - name: indication
    type: str
    required: true
  - name: technique
    type: str
    required: true
  - name: findings
    type: list
    item:
      type: object
      fields:
        - name: category
          type: enum
          values: ["nodule", "lymphadenopathy", "effusion", "other"]
        - name: description
          type: str
        - name: size_mm
          type: float
          required: false
  - name: impression
    type: str
    required: true
"""


class TestTemplateCompiler:

    def test_compile_from_yaml_string(self):
        from mosaicx.schemas.template_compiler import compile_template
        Model = compile_template(SAMPLE_TEMPLATE_YAML)
        assert Model.__name__ == "ChestCTReport"

    def test_compiled_model_has_fields(self):
        from mosaicx.schemas.template_compiler import compile_template
        Model = compile_template(SAMPLE_TEMPLATE_YAML)
        fields = set(Model.model_fields.keys())
        assert "indication" in fields
        assert "technique" in fields
        assert "findings" in fields
        assert "impression" in fields

    def test_compiled_model_validates(self):
        from mosaicx.schemas.template_compiler import compile_template
        Model = compile_template(SAMPLE_TEMPLATE_YAML)
        instance = Model(
            indication="Cough",
            technique="CT chest with contrast",
            findings=[{"category": "nodule", "description": "5mm RUL nodule"}],
            impression="Pulmonary nodule, recommend follow-up.",
        )
        assert instance.indication == "Cough"
        assert len(instance.findings) == 1

    def test_compile_from_file(self, tmp_path):
        from mosaicx.schemas.template_compiler import compile_template_file
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(SAMPLE_TEMPLATE_YAML)
        Model = compile_template_file(yaml_file)
        assert Model.__name__ == "ChestCTReport"

    def test_required_field_validation(self):
        from mosaicx.schemas.template_compiler import compile_template
        Model = compile_template(SAMPLE_TEMPLATE_YAML)
        with pytest.raises(Exception):  # Pydantic ValidationError
            Model(indication="Cough")  # missing technique and impression

    def test_parse_template_metadata(self):
        from mosaicx.schemas.template_compiler import parse_template
        meta = parse_template(SAMPLE_TEMPLATE_YAML)
        assert meta.name == "ChestCTReport"
        assert meta.description == "Structured chest CT report"
        assert len(meta.sections) == 4

    def test_parse_template_adds_absence_value_for_optional_enum(self):
        from mosaicx.schemas.template_compiler import parse_template

        yaml_text = """\
name: CervicalLite
sections:
  - name: findings
    type: object
    fields:
      - name: stenosis
        type: enum
        required: false
        values: ["Mild", "Severe"]
      - name: bulge
        type: enum
        required: false
        values: ["disc", "annular"]
"""
        meta = parse_template(yaml_text)
        findings = meta.sections[0]
        assert findings.fields is not None
        by_name = {f.name: f for f in findings.fields}
        assert by_name["stenosis"].values == ["Mild", "Severe", "None"]
        assert by_name["bulge"].values == ["disc", "annular", "none"]

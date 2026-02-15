# tests/test_schema_gen.py
"""Tests for the schema generator pipeline."""

import pytest
from pydantic import BaseModel


class TestSchemaSpec:
    """Test SchemaSpec model itself."""

    def test_schema_spec_construction(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec

        spec = SchemaSpec(
            class_name="PatientRecord",
            description="A patient record",
            fields=[
                FieldSpec(name="name", type="str", description="Patient name", required=True),
                FieldSpec(name="age", type="int", description="Age in years", required=True),
                FieldSpec(name="diagnosis", type="str", description="Diagnosis", required=False),
            ],
        )
        assert spec.class_name == "PatientRecord"
        assert len(spec.fields) == 3

    def test_compile_to_model(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, compile_schema

        spec = SchemaSpec(
            class_name="TestModel",
            description="Test",
            fields=[
                FieldSpec(name="name", type="str", description="Name", required=True),
                FieldSpec(name="score", type="float", description="Score", required=False),
            ],
        )
        Model = compile_schema(spec)
        assert Model.__name__ == "TestModel"
        instance = Model(name="Alice")
        assert instance.name == "Alice"
        assert instance.score is None

    def test_compile_with_list_field(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, compile_schema

        spec = SchemaSpec(
            class_name="ReportModel",
            description="Report",
            fields=[
                FieldSpec(name="findings", type="list[str]", description="List of findings", required=True),
            ],
        )
        Model = compile_schema(spec)
        instance = Model(findings=["nodule", "effusion"])
        assert len(instance.findings) == 2

    def test_compile_with_enum_field(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, compile_schema

        spec = SchemaSpec(
            class_name="GenderModel",
            description="Gender",
            fields=[
                FieldSpec(
                    name="gender",
                    type="enum",
                    description="Gender",
                    required=True,
                    enum_values=["male", "female", "other"],
                ),
            ],
        )
        Model = compile_schema(spec)
        instance = Model(gender="male")
        assert instance.gender == "male"


class TestSchemaGeneratorSignature:
    """Test the DSPy signature exists and has correct fields."""

    def test_signature_has_fields(self):
        from mosaicx.pipelines.schema_gen import GenerateSchemaSpec

        sig = GenerateSchemaSpec
        assert "description" in sig.input_fields
        assert "schema_spec" in sig.output_fields

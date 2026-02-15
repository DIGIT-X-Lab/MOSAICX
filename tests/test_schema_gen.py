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


class TestSchemaStorage:
    """Test schema save/load/list operations."""

    def test_save_and_load_schema(self, tmp_path):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, save_schema, load_schema

        spec = SchemaSpec(
            class_name="TestSave",
            description="Test save",
            fields=[
                FieldSpec(name="name", type="str", description="Name", required=True),
            ],
        )
        save_schema(spec, tmp_path)
        loaded = load_schema("TestSave", tmp_path)
        assert loaded.class_name == "TestSave"
        assert len(loaded.fields) == 1
        assert loaded.fields[0].name == "name"

    def test_save_creates_json_file(self, tmp_path):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, save_schema

        spec = SchemaSpec(
            class_name="FileCheck",
            description="Check file",
            fields=[FieldSpec(name="x", type="int", required=True)],
        )
        path = save_schema(spec, tmp_path)
        assert path.name == "FileCheck.json"
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path):
        from mosaicx.pipelines.schema_gen import load_schema

        with pytest.raises(FileNotFoundError):
            load_schema("DoesNotExist", tmp_path)

    def test_list_schemas_empty(self, tmp_path):
        from mosaicx.pipelines.schema_gen import list_schemas

        result = list_schemas(tmp_path)
        assert result == []

    def test_list_schemas_returns_specs(self, tmp_path):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, save_schema, list_schemas

        for name in ["Alpha", "Beta"]:
            save_schema(
                SchemaSpec(
                    class_name=name,
                    description=f"{name} schema",
                    fields=[FieldSpec(name="x", type="str", required=True)],
                ),
                tmp_path,
            )
        result = list_schemas(tmp_path)
        assert len(result) == 2
        names = {s.class_name for s in result}
        assert names == {"Alpha", "Beta"}

    def test_save_to_custom_path(self, tmp_path):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, save_schema

        custom = tmp_path / "custom.json"
        spec = SchemaSpec(
            class_name="Custom",
            description="Custom path",
            fields=[FieldSpec(name="x", type="str", required=True)],
        )
        save_schema(spec, output_path=custom)
        assert custom.exists()


class TestSchemaGeneratorSignature:
    """Test the DSPy signature exists and has correct fields."""

    def test_signature_has_fields(self):
        from mosaicx.pipelines.schema_gen import GenerateSchemaSpec

        sig = GenerateSchemaSpec
        assert "description" in sig.input_fields
        assert "schema_spec" in sig.output_fields

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

    def test_compile_with_list_object_field(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, compile_schema

        spec = SchemaSpec(
            class_name="CervicalSpine",
            description="Nested level findings",
            fields=[
                FieldSpec(
                    name="level_findings",
                    type="list[object]",
                    required=True,
                    fields=[
                        FieldSpec(name="level", type="str", required=True),
                        FieldSpec(name="stenosis", type="enum", required=True, enum_values=["none", "mild", "severe"]),
                    ],
                ),
            ],
        )
        Model = compile_schema(spec)
        instance = Model(level_findings=[{"level": "C5-C6", "stenosis": "mild"}])
        assert instance.level_findings[0].level == "C5-C6"
        assert str(instance.level_findings[0].stenosis.value) == "mild"

    def test_compile_list_object_without_fields_falls_back_to_dict(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, compile_schema

        spec = SchemaSpec(
            class_name="FallbackListObject",
            fields=[
                FieldSpec(name="items", type="list[object]", required=True),
            ],
        )
        Model = compile_schema(spec)
        instance = Model(items=[{"k": "v"}])
        assert instance.items == [{"k": "v"}]


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


class TestSchemaFieldOps:
    """Test direct field manipulation for CLI-flag refinement."""

    def _base_spec(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec

        return SchemaSpec(
            class_name="Test",
            description="Test",
            fields=[
                FieldSpec(name="name", type="str", description="Name", required=True),
                FieldSpec(name="age", type="int", description="Age", required=True),
            ],
        )

    def test_add_field(self):
        from mosaicx.pipelines.schema_gen import add_field

        spec = self._base_spec()
        updated = add_field(spec, "email", "str")
        assert len(updated.fields) == 3
        assert updated.fields[-1].name == "email"
        assert updated.fields[-1].type == "str"

    def test_remove_field(self):
        from mosaicx.pipelines.schema_gen import remove_field

        spec = self._base_spec()
        updated = remove_field(spec, "age")
        assert len(updated.fields) == 1
        assert updated.fields[0].name == "name"

    def test_remove_nonexistent_raises(self):
        from mosaicx.pipelines.schema_gen import remove_field

        spec = self._base_spec()
        with pytest.raises(ValueError, match="not found"):
            remove_field(spec, "nonexistent")

    def test_rename_field(self):
        from mosaicx.pipelines.schema_gen import rename_field

        spec = self._base_spec()
        updated = rename_field(spec, "name", "full_name")
        assert updated.fields[0].name == "full_name"

    def test_rename_nonexistent_raises(self):
        from mosaicx.pipelines.schema_gen import rename_field

        spec = self._base_spec()
        with pytest.raises(ValueError, match="not found"):
            rename_field(spec, "nonexistent", "new_name")


class TestSchemaGeneratorSignature:
    """Test the DSPy signature exists and has correct fields."""

    def test_signature_has_fields(self):
        from mosaicx.pipelines.schema_gen import GenerateSchemaSpec

        sig = GenerateSchemaSpec
        assert "description" in sig.input_fields
        assert "schema_spec" in sig.output_fields


class TestSchemaRefinerSignature:
    """Test the DSPy refinement signature exists and has correct fields."""

    def test_refine_signature_has_fields(self):
        from mosaicx.pipelines.schema_gen import RefineSchemaSpec

        sig = RefineSchemaSpec
        assert "current_schema" in sig.input_fields
        assert "instruction" in sig.input_fields
        assert "refined_schema" in sig.output_fields


class TestSchemaNormalization:
    def test_normalize_schema_spec_sanitizes_identifiers_and_types(self):
        from mosaicx.pipelines.schema_gen import FieldSpec, SchemaSpec, normalize_schema_spec

        spec = SchemaSpec(
            class_name="123 bad schema",
            fields=[
                FieldSpec(name="Patient Name", type="STRING", required=True),
                FieldSpec(name="Patient Name", type="integer", required=True),
                FieldSpec(name="Status", type="enum", enum_values=["", "Active", "active"]),
                FieldSpec(name="Free Form", type="", required=False),
            ],
        )

        normalized = normalize_schema_spec(spec)
        assert normalized.class_name.startswith("Schema")
        names = [f.name for f in normalized.fields]
        assert names[0] == "patient_name"
        assert names[1] == "patient_name_2"
        assert names[2] == "status"
        assert names[3] == "free_form"
        assert normalized.fields[0].type == "str"
        assert normalized.fields[1].type == "int"
        assert normalized.fields[2].enum_values == ["Active"]
        assert normalized.fields[3].type == "str"

    def test_validate_schema_spec_reports_missing_fields(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, validate_schema_spec

        spec = SchemaSpec(class_name="ValidName", fields=[])
        issues = validate_schema_spec(spec)
        assert any("no fields" in issue.lower() for issue in issues)

    def test_validate_schema_spec_flags_non_canonical_types(self):
        from mosaicx.pipelines.schema_gen import FieldSpec, SchemaSpec, validate_schema_spec

        spec = SchemaSpec(
            class_name="MySchema",
            fields=[FieldSpec(name="x", type="string", required=True)],
        )
        issues = validate_schema_spec(spec)
        assert any("non-canonical type" in issue.lower() for issue in issues)

    def test_runtime_validate_schema_passes_with_populated_required_fields(self, monkeypatch):
        from types import SimpleNamespace

        from mosaicx.pipelines import extraction as extraction_mod
        from mosaicx.pipelines.schema_gen import (
            FieldSpec,
            SchemaSpec,
            _runtime_validate_schema,
        )

        class FakeExtractor:
            def __init__(self, output_schema=None):  # noqa: ANN001
                self.output_schema = output_schema

            def __call__(self, document_text: str):  # noqa: ARG002
                return SimpleNamespace(extracted={"patient_id": "1234", "study_date": "2013-01-01"})

        monkeypatch.setattr(extraction_mod, "DocumentExtractor", FakeExtractor)
        spec = SchemaSpec(
            class_name="RuntimeCheck",
            fields=[
                FieldSpec(name="patient_id", type="str", required=True),
                FieldSpec(name="study_date", type="str", required=True),
            ],
        )
        ok, issues = _runtime_validate_schema(spec, document_text="report text")
        assert ok is True
        assert issues == []

    def test_runtime_validate_schema_flags_missing_required_fields(self, monkeypatch):
        from types import SimpleNamespace

        from mosaicx.pipelines import extraction as extraction_mod
        from mosaicx.pipelines.schema_gen import (
            FieldSpec,
            SchemaSpec,
            _runtime_validate_schema,
        )

        class FakeExtractor:
            def __init__(self, output_schema=None):  # noqa: ANN001
                self.output_schema = output_schema

            def __call__(self, document_text: str):  # noqa: ARG002
                return SimpleNamespace(extracted={"patient_id": ""})

        monkeypatch.setattr(extraction_mod, "DocumentExtractor", FakeExtractor)
        spec = SchemaSpec(
            class_name="RuntimeCheck",
            fields=[
                FieldSpec(name="patient_id", type="str", required=True),
                FieldSpec(name="study_date", type="str", required=True),
            ],
        )
        ok, issues = _runtime_validate_schema(spec, document_text="report text")
        assert ok is False
        assert any("runtime_missing_required_fields" in issue for issue in issues)

    def test_build_synthetic_runtime_probe_contains_schema_payload(self):
        from mosaicx.pipelines.schema_gen import (
            FieldSpec,
            SchemaSpec,
            _build_synthetic_runtime_probe_text,
        )

        spec = SchemaSpec(
            class_name="ProbeSchema",
            fields=[
                FieldSpec(name="patient_id", type="str", required=True),
                FieldSpec(name="severity", type="enum", enum_values=["mild", "severe"], required=True),
                FieldSpec(
                    name="level_findings",
                    type="list[object]",
                    required=True,
                    fields=[
                        FieldSpec(name="level", type="str", required=True),
                        FieldSpec(name="stenosis", type="enum", enum_values=["none", "mild"], required=True),
                    ],
                ),
            ],
        )

        probe = _build_synthetic_runtime_probe_text(
            spec,
            description="C-spine MRI report",
            example_text="Disc bulge at C3-C4.",
        )
        assert "Synthetic document for schema runtime validation." in probe
        assert '"patient_id"' in probe
        assert '"severity"' in probe
        assert '"level_findings"' in probe
        assert "Schema description: C-spine MRI report" in probe

    def test_schema_generator_runtime_dryrun_uses_synthetic_probe_when_no_document(self, monkeypatch):
        from types import SimpleNamespace

        import mosaicx.pipelines.schema_gen as schema_mod
        from mosaicx.pipelines.schema_gen import FieldSpec, SchemaGenerator, SchemaSpec

        spec = SchemaSpec(
            class_name="RuntimeProbeSchema",
            fields=[FieldSpec(name="patient_id", type="str", required=True)],
        )
        generator = SchemaGenerator()
        generator.generate = lambda **_: SimpleNamespace(schema_spec=spec)
        generator.repair = lambda **_: SimpleNamespace(repaired_schema=spec)

        captured: dict[str, str] = {}

        def fake_runtime_validate(candidate, *, document_text, missing_required_threshold=0.5):  # noqa: ANN001
            captured["document_text"] = str(document_text)
            captured["threshold"] = str(missing_required_threshold)
            return True, []

        monkeypatch.setattr(schema_mod, "_runtime_validate_schema", fake_runtime_validate)

        out = generator.forward(
            description="Simple patient ID extraction",
            document_text="",
            runtime_dryrun=True,
        )
        assert out.runtime_dryrun_used is True
        assert "Synthetic document for schema runtime validation." in captured["document_text"]
        assert '"patient_id"' in captured["document_text"]

    def test_schema_generator_runtime_dryrun_prefers_real_document_text(self, monkeypatch):
        from types import SimpleNamespace

        import mosaicx.pipelines.schema_gen as schema_mod
        from mosaicx.pipelines.schema_gen import FieldSpec, SchemaGenerator, SchemaSpec

        spec = SchemaSpec(
            class_name="RuntimeProbeSchema",
            fields=[FieldSpec(name="patient_id", type="str", required=True)],
        )
        generator = SchemaGenerator()
        generator.generate = lambda **_: SimpleNamespace(schema_spec=spec)
        generator.repair = lambda **_: SimpleNamespace(repaired_schema=spec)

        captured: dict[str, str] = {}

        def fake_runtime_validate(candidate, *, document_text, missing_required_threshold=0.5):  # noqa: ANN001
            captured["document_text"] = str(document_text)
            return True, []

        monkeypatch.setattr(schema_mod, "_runtime_validate_schema", fake_runtime_validate)

        out = generator.forward(
            description="Simple patient ID extraction",
            document_text="Ground truth source text",
            runtime_dryrun=True,
        )
        assert out.runtime_dryrun_used is True
        assert captured["document_text"] == "Ground truth source text"

    def test_assess_schema_semantic_granularity_flags_flat_schema_for_levelled_text(self):
        from mosaicx.pipelines.schema_gen import (
            FieldSpec,
            SchemaSpec,
            assess_schema_semantic_granularity,
        )

        document_text = """
Findings:
C2-3: mild disc desiccation.
C3-4: broad-based bulge with mild stenosis.
C4-5: no significant narrowing.
C5-6: moderate foraminal narrowing.
C6-7: severe bilateral foraminal narrowing.
Impression:
1. Mild bulging at C3-4.
2. Moderate narrowing at C5-6.
3. Severe bilateral narrowing at C6-7.
"""
        flat = SchemaSpec(
            class_name="FlatSpineSchema",
            fields=[
                FieldSpec(name="findings", type="list[str]", required=True),
                FieldSpec(name="impression", type="list[str]", required=True),
            ],
        )
        score, issues = assess_schema_semantic_granularity(flat, document_text=document_text)
        assert score < 0.55
        assert any("lacks_list_object_structure" in issue for issue in issues)

    def test_assess_schema_semantic_granularity_rewards_structured_schema_for_levelled_text(self):
        from mosaicx.pipelines.schema_gen import (
            FieldSpec,
            SchemaSpec,
            assess_schema_semantic_granularity,
        )

        document_text = """
Findings:
C2-3: mild disc desiccation.
C3-4: broad-based bulge with mild stenosis.
C4-5: no significant narrowing.
C5-6: moderate foraminal narrowing.
C6-7: severe bilateral foraminal narrowing.
"""
        structured = SchemaSpec(
            class_name="StructuredSpineSchema",
            fields=[
                FieldSpec(
                    name="level_findings",
                    type="list[object]",
                    required=True,
                    fields=[
                        FieldSpec(name="level", type="str", required=True),
                        FieldSpec(name="severity", type="enum", enum_values=["none", "mild", "moderate", "severe"], required=True),
                    ],
                ),
                FieldSpec(name="impression", type="list[str]", required=False),
            ],
        )
        score, issues = assess_schema_semantic_granularity(
            structured,
            document_text=document_text,
        )
        assert score >= 0.55
        assert not any("lacks_list_object_structure" in issue for issue in issues)

    def test_assess_schema_semantic_granularity_flags_flat_schema_for_numbered_lesions(self):
        from mosaicx.pipelines.schema_gen import (
            FieldSpec,
            SchemaSpec,
            assess_schema_semantic_granularity,
        )

        document_text = """
Impression:
1. Liver lesion in segment IV, size 12 mm, likely benign.
2. Right adrenal lesion, size 18 mm, indeterminate.
3. Left iliac node, size 9 mm, suspicious.
"""
        flat = SchemaSpec(
            class_name="FlatLesionSchema",
            fields=[
                FieldSpec(name="impression", type="list[str]", required=True),
                FieldSpec(name="summary", type="str", required=False),
            ],
        )
        score, issues = assess_schema_semantic_granularity(flat, document_text=document_text)
        assert score < 0.6
        assert any("numbered_structures" in issue for issue in issues)

    def test_assess_schema_semantic_granularity_rewards_structured_lesion_schema(self):
        from mosaicx.pipelines.schema_gen import (
            FieldSpec,
            SchemaSpec,
            assess_schema_semantic_granularity,
        )

        document_text = """
Impression:
1. Liver lesion in segment IV, size 12 mm, likely benign.
2. Right adrenal lesion, size 18 mm, indeterminate.
3. Left iliac node, size 9 mm, suspicious.
"""
        structured = SchemaSpec(
            class_name="StructuredLesionSchema",
            fields=[
                FieldSpec(
                    name="lesions",
                    type="list[object]",
                    required=True,
                    fields=[
                        FieldSpec(name="location", type="str", required=True),
                        FieldSpec(name="size_mm", type="float", required=True),
                        FieldSpec(name="status", type="enum", enum_values=["benign", "indeterminate", "suspicious"], required=True),
                    ],
                ),
            ],
        )
        score, issues = assess_schema_semantic_granularity(
            structured,
            document_text=document_text,
        )
        assert score >= 0.6
        assert not any("numbered_structures" in issue for issue in issues)

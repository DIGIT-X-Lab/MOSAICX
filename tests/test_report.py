# tests/test_report.py
"""Tests for structured reporting: completeness scoring, report orchestration,
template resolution, and CLI commands."""

from __future__ import annotations

from typing import Any, Optional

import pytest
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic models for testing
# ---------------------------------------------------------------------------


class FullyRequiredReport(BaseModel):
    """All fields required -- no optional fields."""
    indication: str
    findings: str
    impression: str


class MixedReport(BaseModel):
    """Mix of required and optional fields."""
    indication: str
    findings: str
    impression: str
    technique: Optional[str] = None
    clinical_history: Optional[str] = None


class AllOptionalReport(BaseModel):
    """All fields optional."""
    notes: Optional[str] = None
    addendum: Optional[str] = None
    followup: Optional[str] = None


class EmptyModel(BaseModel):
    """Model with no fields at all."""
    pass


# ---------------------------------------------------------------------------
# compute_report_completeness tests
# ---------------------------------------------------------------------------


class TestComputeReportCompleteness:
    """Test the template-aware completeness scoring."""

    def test_all_required_fields_filled(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = FullyRequiredReport(
            indication="Chest pain",
            findings="5mm nodule RUL",
            impression="Follow-up CT in 3 months",
        )
        result = compute_report_completeness(model, source_text="Patient reports chest pain.")
        assert result.overall == 1.0
        assert result.required_coverage == 1.0
        assert result.field_coverage == 1.0
        assert result.filled_fields == 3
        assert result.total_fields == 3
        assert result.missing_required == []

    def test_all_required_fields_empty(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = FullyRequiredReport(indication="", findings="", impression="")
        result = compute_report_completeness(model, source_text="Some source text")
        assert result.overall == 0.0
        assert result.required_coverage == 0.0
        assert result.filled_fields == 0
        assert len(result.missing_required) == 3

    def test_mixed_required_optional_all_filled(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = MixedReport(
            indication="Cough",
            findings="Clear lungs",
            impression="Normal",
            technique="CT",
            clinical_history="Chronic cough",
        )
        result = compute_report_completeness(model, source_text="Patient with cough")
        assert result.overall == 1.0
        assert result.required_coverage == 1.0
        assert result.optional_coverage == 1.0
        assert result.filled_fields == 5
        assert result.missing_required == []

    def test_mixed_required_filled_optional_empty(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = MixedReport(
            indication="Cough",
            findings="Clear lungs",
            impression="Normal",
        )
        result = compute_report_completeness(model, source_text="Patient with cough")
        assert result.required_coverage == 1.0
        assert result.optional_coverage == 0.0
        assert result.filled_fields == 3
        assert result.total_fields == 5
        assert result.missing_required == []
        # Overall should reflect that required fields are filled but optional are not
        # With default weights: 3 required * 1.0 + 2 optional * 0.3 = total 3.6
        # filled: 3 * 1.0 = 3.0, so overall ~= 3.0 / 3.6 ≈ 0.833
        assert 0.8 < result.overall < 0.9

    def test_missing_one_required(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = MixedReport(
            indication="",
            findings="Clear lungs",
            impression="Normal",
        )
        result = compute_report_completeness(model, source_text="Patient with cough")
        assert result.required_coverage == pytest.approx(2 / 3)
        assert result.missing_required == ["indication"]

    def test_all_optional_all_filled(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = AllOptionalReport(notes="OK", addendum="None", followup="3 months")
        result = compute_report_completeness(model, source_text="Some text")
        assert result.overall == 1.0
        assert result.required_coverage == 1.0  # no required fields -> default 1.0
        assert result.optional_coverage == 1.0

    def test_all_optional_none_filled(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = AllOptionalReport()
        result = compute_report_completeness(model, source_text="Some text")
        assert result.overall == 0.0
        assert result.optional_coverage == 0.0
        assert result.missing_required == []

    def test_empty_model(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = EmptyModel()
        result = compute_report_completeness(model, source_text="Some text")
        assert result.overall == 0.0
        assert result.total_fields == 0

    def test_per_field_breakdown(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = MixedReport(
            indication="Cough",
            findings="",
            impression="Normal",
            technique="CT",
        )
        result = compute_report_completeness(model, source_text="Text")
        assert len(result.fields) == 5

        # Check individual fields
        by_name = {f.name: f for f in result.fields}
        assert by_name["indication"].filled is True
        assert by_name["indication"].required is True
        assert by_name["findings"].filled is False
        assert by_name["findings"].required is True
        assert by_name["technique"].filled is True
        assert by_name["technique"].required is False
        assert by_name["clinical_history"].filled is False
        assert by_name["clinical_history"].required is False

    def test_information_density_included(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = FullyRequiredReport(
            indication="Chest pain",
            findings="Nodule",
            impression="Follow-up",
        )
        result = compute_report_completeness(
            model,
            source_text="Very long source text with many words and details about the patient",
        )
        assert 0.0 <= result.information_density <= 1.0

    def test_custom_weights(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = MixedReport(
            indication="Cough",
            findings="Clear",
            impression="Normal",
        )
        # Equal weights
        result = compute_report_completeness(
            model, source_text="Text",
            required_weight=1.0, optional_weight=1.0,
        )
        assert result.overall == pytest.approx(3 / 5)

    def test_model_class_override(self):
        from mosaicx.evaluation.completeness import compute_report_completeness

        model = MixedReport(
            indication="Cough",
            findings="Clear",
            impression="Normal",
        )
        # Use FullyRequiredReport as the class to inspect (all 3 filled fields
        # are now considered required -- should give 100%)
        result = compute_report_completeness(
            model, source_text="Text", model_class=FullyRequiredReport,
        )
        assert result.overall == 1.0
        assert result.required_coverage == 1.0


# ---------------------------------------------------------------------------
# FieldCompleteness / ReportCompleteness dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_field_completeness_creation(self):
        from mosaicx.evaluation.completeness import FieldCompleteness

        fc = FieldCompleteness(
            name="indication", filled=True, required=True, value_summary="'Cough'"
        )
        assert fc.name == "indication"
        assert fc.filled is True

    def test_report_completeness_as_dict(self):
        from dataclasses import asdict

        from mosaicx.evaluation.completeness import ReportCompleteness

        rc = ReportCompleteness(
            overall=0.85,
            field_coverage=0.8,
            required_coverage=1.0,
            optional_coverage=0.5,
            information_density=0.6,
            total_fields=5,
            filled_fields=4,
            missing_required=[],
        )
        d = asdict(rc)
        assert d["overall"] == 0.85
        assert isinstance(d["missing_required"], list)


# ---------------------------------------------------------------------------
# Template registry mode field tests
# ---------------------------------------------------------------------------


class TestRegistryModeField:
    def test_chest_ct_has_radiology_mode(self):
        from mosaicx.schemas.radreport.registry import get_template

        t = get_template("chest_ct")
        assert t is not None
        assert t.mode == "radiology"

    def test_brain_mri_has_radiology_mode(self):
        from mosaicx.schemas.radreport.registry import get_template

        t = get_template("brain_mri")
        assert t is not None
        assert t.mode == "radiology"

    def test_all_templates_have_mode(self):
        from mosaicx.schemas.radreport.registry import list_templates

        for t in list_templates():
            assert t.mode is not None, f"Template {t.name!r} has no mode"

    def test_unknown_template_returns_none(self):
        from mosaicx.schemas.radreport.registry import get_template

        t = get_template("nonexistent_xyz")
        assert t is None


# ---------------------------------------------------------------------------
# Report module: resolve_template and detect_mode
# ---------------------------------------------------------------------------


class TestDetectMode:
    def test_detect_mode_chest_ct(self):
        from mosaicx.report import detect_mode

        assert detect_mode("chest_ct") == "radiology"

    def test_detect_mode_unknown(self):
        from mosaicx.report import detect_mode

        assert detect_mode("unknown_thing") is None

    def test_detect_mode_none(self):
        from mosaicx.report import detect_mode

        assert detect_mode(None) is None


class TestResolveTemplate:
    def test_resolve_unknown_template_raises(self):
        from mosaicx.report import resolve_template

        with pytest.raises(ValueError, match="not found"):
            resolve_template(template="nonexistent_xyz_template")

    def test_resolve_none_returns_none(self):
        from mosaicx.report import resolve_template

        model, name = resolve_template()
        assert model is None
        assert name is None

    def test_resolve_builtin_template(self):
        """Built-in RDES template resolves to (None, name) when no YAML file exists."""
        from mosaicx.report import resolve_template

        # chest_ct is registered but may not have a YAML file
        model, name = resolve_template(template="chest_ct")
        assert name == "chest_ct"
        # model may be None if no YAML file exists for this template

    def test_resolve_saved_schema_fallback(self, tmp_path):
        """Template resolution falls back to saved schemas."""
        from mosaicx.report import resolve_template

        # Create a saved schema
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        (schema_dir / "TestModel.json").write_text(
            '{"class_name":"TestModel","description":"Test",'
            '"fields":[{"name":"x","type":"str","description":"x",'
            '"required":true,"enum_values":null}]}'
        )

        model, name = resolve_template(
            template="TestModel", schema_dir=schema_dir
        )
        assert name == "TestModel"
        assert model is not None
        assert "x" in model.model_fields

    def test_resolve_builtin_takes_priority_over_saved(self, tmp_path):
        """Built-in templates take priority over saved schemas with same name."""
        from mosaicx.report import resolve_template

        # Create a saved schema with a name that collides with a built-in
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        (schema_dir / "chest_ct.json").write_text(
            '{"class_name":"chest_ct","description":"Custom chest CT",'
            '"fields":[{"name":"custom","type":"str","description":"custom",'
            '"required":true,"enum_values":null}]}'
        )

        model, name = resolve_template(
            template="chest_ct", schema_dir=schema_dir
        )
        # Should resolve to built-in (not saved schema)
        assert name == "chest_ct"
        assert model is not None
        # Built-in has "indication" field, not "custom"
        assert "indication" in model.model_fields

    def test_resolve_error_message_suggests_template_list(self):
        """Error message suggests 'mosaicx template list'."""
        from mosaicx.report import resolve_template

        with pytest.raises(ValueError, match="template list"):
            resolve_template(template="nonexistent_xyz")

    def test_resolve_yaml_file_first(self, tmp_path):
        """YAML file takes priority over saved schema with same name."""
        from mosaicx.report import resolve_template

        # Create a YAML template
        yaml_file = tmp_path / "my_template.yaml"
        yaml_file.write_text(
            "name: MyTemplate\n"
            "description: Test\n"
            "sections:\n"
            "  - name: field_a\n"
            "    type: str\n"
            "    required: true\n"
        )

        model, name = resolve_template(template=str(yaml_file))
        assert name == "my_template"
        assert model is not None

    def test_resolve_none_template_returns_none_none(self):
        """Passing None returns (None, None)."""
        from mosaicx.report import resolve_template

        model, name = resolve_template(template=None)
        assert model is None
        assert name is None


# ---------------------------------------------------------------------------
# render_completeness display
# ---------------------------------------------------------------------------


class TestRenderCompleteness:
    def test_render_completeness_runs(self):
        """Verify render_completeness doesn't crash."""
        from io import StringIO

        from rich.console import Console

        from mosaicx.cli_display import render_completeness

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=80)

        completeness_dict = {
            "overall": 0.85,
            "field_coverage": 0.8,
            "required_coverage": 1.0,
            "optional_coverage": 0.5,
            "information_density": 0.6,
            "total_fields": 5,
            "filled_fields": 4,
            "missing_required": [],
            "fields": [
                {"name": "indication", "filled": True, "required": True, "value_summary": "'Cough'"},
                {"name": "findings", "filled": True, "required": True, "value_summary": "'Clear'"},
                {"name": "impression", "filled": True, "required": True, "value_summary": "'Normal'"},
                {"name": "technique", "filled": True, "required": False, "value_summary": "'CT'"},
                {"name": "clinical_history", "filled": False, "required": False, "value_summary": "---"},
            ],
        }

        render_completeness(completeness_dict, test_console)
        output = buf.getvalue()
        assert "85%" in output
        assert "4/5" in output

    def test_render_completeness_with_missing_required(self):
        """Verify missing required fields are shown."""
        from io import StringIO

        from rich.console import Console

        from mosaicx.cli_display import render_completeness

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=True, width=80)

        completeness_dict = {
            "overall": 0.5,
            "field_coverage": 0.5,
            "required_coverage": 0.5,
            "optional_coverage": 0.0,
            "information_density": 0.3,
            "total_fields": 4,
            "filled_fields": 2,
            "missing_required": ["findings", "impression"],
            "fields": [
                {"name": "indication", "filled": True, "required": True, "value_summary": "'Cough'"},
                {"name": "findings", "filled": False, "required": True, "value_summary": "---"},
                {"name": "impression", "filled": False, "required": True, "value_summary": "---"},
                {"name": "technique", "filled": True, "required": False, "value_summary": "'CT'"},
            ],
        }

        render_completeness(completeness_dict, test_console)
        output = buf.getvalue()
        assert "findings" in output
        assert "impression" in output
        assert "Missing required" in output


# ---------------------------------------------------------------------------
# _find_primary_model tests
# ---------------------------------------------------------------------------


class _FakePrediction:
    """Lightweight stand-in for dspy.Prediction (keys + attribute access)."""

    def __init__(self, **kwargs: Any):
        self._fields = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def keys(self):
        return self._fields.keys()


class TestFindPrimaryModel:
    def test_finds_basemodel_in_prediction(self):
        from mosaicx.report import _find_primary_model

        model = FullyRequiredReport(
            indication="Chest pain", findings="Nodule", impression="Follow-up"
        )
        pred = _FakePrediction(
            exam_type="chest_ct",
            sections=model,
            modality="CT",
        )
        result = _find_primary_model(pred)
        assert result is model

    def test_returns_none_when_no_models(self):
        from mosaicx.report import _find_primary_model

        pred = _FakePrediction(exam_type="chest_ct", modality="CT")
        assert _find_primary_model(pred) is None

    def test_returns_first_model_when_multiple(self):
        from mosaicx.report import _find_primary_model

        first = FullyRequiredReport(
            indication="A", findings="B", impression="C"
        )
        second = MixedReport(
            indication="X", findings="Y", impression="Z"
        )
        pred = _FakePrediction(
            exam_type="chest_ct",
            sections=first,
            details=second,
        )
        result = _find_primary_model(pred)
        assert result is first

    def test_skips_lists_of_models(self):
        from mosaicx.report import _find_primary_model

        model = FullyRequiredReport(
            indication="A", findings="B", impression="C"
        )
        pred = _FakePrediction(
            items=[model],
            primary=model,
        )
        # Lists are not BaseModel instances, so it should skip them
        # and find the direct BaseModel at 'primary'
        result = _find_primary_model(pred)
        assert result is model


# ---------------------------------------------------------------------------
# ReportResult dataclass
# ---------------------------------------------------------------------------


class TestReportResult:
    def test_report_result_creation(self):
        from mosaicx.report import ReportResult

        r = ReportResult(
            extracted={"indication": "Cough"},
            completeness={"overall": 0.9},
            template_name="chest_ct",
            mode_used="radiology",
        )
        assert r.template_name == "chest_ct"
        assert r.mode_used == "radiology"
        assert r.extracted["indication"] == "Cough"

    def test_report_result_as_dict(self):
        from dataclasses import asdict

        from mosaicx.report import ReportResult

        r = ReportResult(
            extracted={"x": 1},
            completeness={"overall": 0.5},
        )
        d = asdict(r)
        assert d["extracted"] == {"x": 1}
        assert d["completeness"]["overall"] == 0.5
        assert d["template_name"] is None


# ---------------------------------------------------------------------------
# Built-in YAML template resolution
# ---------------------------------------------------------------------------


class TestBuiltinYamlTemplates:
    """Tests for resolving built-in YAML templates (Phase 2)."""

    def test_builtin_yaml_resolves_to_model(self):
        """Built-in YAML templates resolve to a compiled Pydantic model."""
        from mosaicx.report import resolve_template

        model, name = resolve_template(template="chest_ct")
        assert name == "chest_ct"
        assert model is not None
        # Should have sections from the YAML file
        assert "indication" in model.model_fields
        assert "impression" in model.model_fields

    def test_all_builtin_templates_compile(self):
        """All 11 built-in YAML templates compile successfully."""
        from mosaicx.report import _find_builtin_template_yaml
        from mosaicx.schemas.radreport.registry import list_templates
        from mosaicx.schemas.template_compiler import compile_template_file

        templates = list_templates()
        for tpl in templates:
            path = _find_builtin_template_yaml(tpl.name)
            assert path is not None, f"No YAML file for {tpl.name}"
            model = compile_template_file(path)
            assert model is not None, f"Failed to compile {tpl.name}"

    def test_detect_mode_from_builtin_yaml(self):
        """detect_mode reads mode from built-in YAML template."""
        from mosaicx.report import detect_mode

        mode = detect_mode("chest_ct")
        assert mode == "radiology"

    def test_detect_mode_from_user_yaml(self, tmp_path, monkeypatch):
        """detect_mode reads mode from user YAML template."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            tpl_dir = tmp_path / "templates"
            tpl_dir.mkdir()
            (tpl_dir / "custom.yaml").write_text(
                "name: Custom\n"
                "description: test\n"
                "mode: pathology\n"
                "sections:\n"
                "  - name: x\n"
                "    type: str\n"
                "    required: true\n"
            )
            from mosaicx.report import detect_mode

            mode = detect_mode("custom")
            assert mode == "pathology"
        finally:
            get_config.cache_clear()

    def test_resolve_user_template_yaml(self, tmp_path, monkeypatch):
        """User templates in ~/.mosaicx/templates/ are resolved."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config

        get_config.cache_clear()
        try:
            tpl_dir = tmp_path / "templates"
            tpl_dir.mkdir()
            (tpl_dir / "my_tpl.yaml").write_text(
                "name: MyTemplate\n"
                "description: Test\n"
                "sections:\n"
                "  - name: result\n"
                "    type: str\n"
                "    required: true\n"
            )
            from mosaicx.report import resolve_template

            model, name = resolve_template(template="my_tpl")
            assert name == "my_tpl"
            assert model is not None
            assert "result" in model.model_fields
        finally:
            get_config.cache_clear()


# ---------------------------------------------------------------------------
# SchemaSpec-to-YAML converter tests
# ---------------------------------------------------------------------------


class TestSchemaSpecToYaml:
    """Tests for the schema_spec_to_template_yaml converter."""

    def test_basic_conversion(self):
        from mosaicx.pipelines.schema_gen import FieldSpec as GenFieldSpec
        from mosaicx.pipelines.schema_gen import SchemaSpec
        from mosaicx.schemas.template_compiler import schema_spec_to_template_yaml

        spec = SchemaSpec(
            class_name="TestReport",
            description="A test report",
            fields=[
                GenFieldSpec(name="findings", type="str", description="Findings", required=True),
                GenFieldSpec(name="count", type="int", description="Count", required=False),
            ],
        )
        yaml_str = schema_spec_to_template_yaml(spec)
        assert "name: TestReport" in yaml_str
        assert "findings" in yaml_str
        assert "count" in yaml_str

    def test_list_type_conversion(self):
        from mosaicx.pipelines.schema_gen import FieldSpec as GenFieldSpec
        from mosaicx.pipelines.schema_gen import SchemaSpec
        from mosaicx.schemas.template_compiler import schema_spec_to_template_yaml

        spec = SchemaSpec(
            class_name="Test",
            fields=[
                GenFieldSpec(name="items", type="list[str]", description="List of items", required=True),
            ],
        )
        yaml_str = schema_spec_to_template_yaml(spec)
        assert "type: list" in yaml_str
        assert "type: str" in yaml_str  # item type

    def test_enum_type_conversion(self):
        from mosaicx.pipelines.schema_gen import FieldSpec as GenFieldSpec
        from mosaicx.pipelines.schema_gen import SchemaSpec
        from mosaicx.schemas.template_compiler import schema_spec_to_template_yaml

        spec = SchemaSpec(
            class_name="Test",
            fields=[
                GenFieldSpec(
                    name="severity", type="enum", description="Level",
                    required=True, enum_values=["mild", "moderate", "severe"],
                ),
            ],
        )
        yaml_str = schema_spec_to_template_yaml(spec)
        assert "type: enum" in yaml_str
        assert "mild" in yaml_str

    def test_mode_embedded(self):
        from mosaicx.pipelines.schema_gen import FieldSpec as GenFieldSpec
        from mosaicx.pipelines.schema_gen import SchemaSpec
        from mosaicx.schemas.template_compiler import schema_spec_to_template_yaml

        spec = SchemaSpec(
            class_name="Test",
            fields=[
                GenFieldSpec(name="x", type="str", description="x", required=True),
            ],
        )
        yaml_str = schema_spec_to_template_yaml(spec, mode="radiology")
        assert "mode: radiology" in yaml_str

    def test_roundtrip_compiles(self):
        """Generated YAML can be compiled back into a Pydantic model."""
        from mosaicx.pipelines.schema_gen import FieldSpec as GenFieldSpec
        from mosaicx.pipelines.schema_gen import SchemaSpec
        from mosaicx.schemas.template_compiler import (
            compile_template,
            schema_spec_to_template_yaml,
        )

        spec = SchemaSpec(
            class_name="RoundTrip",
            description="Test round-trip",
            fields=[
                GenFieldSpec(name="findings", type="str", description="f", required=True),
                GenFieldSpec(name="score", type="float", description="s", required=False),
                GenFieldSpec(name="tags", type="list[str]", description="t", required=False),
            ],
        )
        yaml_str = schema_spec_to_template_yaml(spec, mode="radiology")
        model = compile_template(yaml_str)
        assert model.__name__ == "RoundTrip"
        assert "findings" in model.model_fields
        assert "score" in model.model_fields
        assert "tags" in model.model_fields

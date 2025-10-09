"""
Schema registry unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mosaicx.schema.registry as registry_module


@pytest.fixture()
def registry_env(tmp_path, monkeypatch):
    """Provide an isolated registry environment with temporary directories."""

    user_dir = tmp_path / "user"
    templates_dir = tmp_path / "templates"
    python_dir = templates_dir / "python"
    for root in (user_dir, templates_dir, python_dir):
        root.mkdir(parents=True, exist_ok=True)

    registry_path = tmp_path / "registry.json"

    monkeypatch.setattr(registry_module, "USER_SCHEMA_DIR", user_dir)
    monkeypatch.setattr(registry_module, "PACKAGE_SCHEMA_TEMPLATES_DIR", templates_dir)
    monkeypatch.setattr(registry_module, "PACKAGE_SCHEMA_TEMPLATES_PY_DIR", python_dir)
    monkeypatch.setattr(registry_module, "SCHEMA_REGISTRY_PATH", registry_path)

    registry = registry_module.SchemaRegistry(registry_path)
    monkeypatch.setattr(registry_module, "_registry", registry)
    return registry, user_dir, python_dir


def _write_schema(path: Path, class_name: str = "PatientRecord") -> None:
    """Create a minimal Pydantic schema file for testing."""

    path.write_text(
        f"""
from pydantic import BaseModel


class {class_name}(BaseModel):
    name: str
""",
        encoding="utf-8",
    )


class TestSchemaRegistry:
    def test_register_schema_records_absolute_paths(self, registry_env) -> None:
        _, user_dir, _ = registry_env
        schema_file = user_dir / "patient_identity.py"
        _write_schema(schema_file, "PatientIdentity")

        schema_id = registry_module.register_schema(
            class_name="PatientIdentity",
            description="Demo schema",
            file_path=schema_file,
            model_used="mistral:latest",
        )

        stored = registry_module.get_schema_by_id(schema_id)
        assert stored is not None
        assert Path(stored["file_path"]) == schema_file.resolve()
        assert stored["scope"] == "user"

    def test_list_schemas_supports_filters(self, registry_env) -> None:
        _, user_dir, _ = registry_env
        schema_a = user_dir / "alpha.py"
        schema_b = user_dir / "beta.py"
        _write_schema(schema_a, "AlphaModel")
        _write_schema(schema_b, "BetaModel")

        registry_module.register_schema("AlphaModel", "First", schema_a, "gpt-oss")
        registry_module.register_schema("BetaModel", "Second", schema_b, "gpt-oss")

        all_items = registry_module.list_schemas()
        assert len(all_items) == 2

        filtered = registry_module.list_schemas(class_name_filter="alpha")
        assert len(filtered) == 1
        assert filtered[0]["class_name"] == "AlphaModel"

    def test_cleanup_removes_missing_files(self, registry_env) -> None:
        _, user_dir, _ = registry_env
        schema_path = user_dir / "temp.py"
        _write_schema(schema_path, "TempModel")
        registry_module.register_schema("TempModel", "Desc", schema_path, "gpt-oss")

        schema_path.unlink()
        removed = registry_module.cleanup_missing_files()
        assert removed == 1
        assert registry_module.list_schemas() == []

    def test_scan_registers_builtin_templates(self, registry_env) -> None:
        _, _, python_dir = registry_env
        builtin_file = python_dir / "template.py"
        _write_schema(builtin_file, "TemplateModel")

        registered = registry_module.scan_and_register_existing_schemas()
        assert registered == 1
        entries = registry_module.list_schemas()
        assert entries
        assert entries[0]["scope"] in {"template", "package"}

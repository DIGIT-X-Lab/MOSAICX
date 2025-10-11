from __future__ import annotations

from pathlib import Path

from mosaicx.schema.registry import SchemaRegistry


def test_register_and_list(tmp_path):
    registry_file = tmp_path / "registry.json"
    reg = SchemaRegistry(registry_path=registry_file)

    schema_path = tmp_path / "schemas" / "demo.py"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text("class Demo: ...", encoding="utf-8")

    schema_id = reg.register_schema(
        class_name="Demo",
        description="Demo schema",
        file_path=schema_path,
        model_used="demo-model",
        temperature=0.1,
    )

    entries = reg.list_schemas()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["id"] == schema_id
    assert entry["file_exists"] is True


def test_cleanup_missing_files(tmp_path):
    registry_file = tmp_path / "registry.json"
    reg = SchemaRegistry(registry_path=registry_file)

    schema_path = tmp_path / "missing.py"
    schema_id = reg.register_schema(
        class_name="Missing",
        description="Missing schema",
        file_path=schema_path,
        model_used="demo",
    )

    removed_count = reg.cleanup_missing_files()
    assert removed_count == 1
    assert reg.get_schema_by_id(schema_id) is None

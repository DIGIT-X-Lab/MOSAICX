# Schema Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add save/load/list/refine operations for LLM-generated schemas so they persist across sessions.

**Architecture:** Schemas are stored as `SchemaSpec` JSON files in `~/.mosaicx/schemas/` (configurable). Save/load/list are pure functions in `schema_gen.py`. Refinement uses a new DSPy signature + module. CLI flags provide quick edits without LLM.

**Tech Stack:** Pydantic, DSPy, Click, Rich, pathlib

---

### Task 1: Add save/load/list functions to schema_gen.py

**Files:**
- Modify: `mosaicx/pipelines/schema_gen.py` (after `compile_schema()`, line 163)
- Test: `tests/test_schema_gen.py`

**Step 1: Write failing tests for save/load/list**

Add to `tests/test_schema_gen.py`:

```python
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
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, save_schema, load_schema

        custom = tmp_path / "custom.json"
        spec = SchemaSpec(
            class_name="Custom",
            description="Custom path",
            fields=[FieldSpec(name="x", type="str", required=True)],
        )
        save_schema(spec, output_path=custom)
        assert custom.exists()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schema_gen.py::TestSchemaStorage -v`
Expected: FAIL — `ImportError: cannot import name 'save_schema'`

**Step 3: Implement save/load/list**

Add to `mosaicx/pipelines/schema_gen.py` after `compile_schema()` (after line 163):

```python
# ---------------------------------------------------------------------------
# Schema storage
# ---------------------------------------------------------------------------

def save_schema(
    spec: SchemaSpec,
    schema_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Save a SchemaSpec as a JSON file.

    Args:
        spec: The schema specification to save.
        schema_dir: Directory to save into (uses {class_name}.json).
        output_path: Explicit file path (overrides schema_dir).

    Returns:
        Path to the saved file.
    """
    if output_path is not None:
        dest = output_path
    elif schema_dir is not None:
        schema_dir.mkdir(parents=True, exist_ok=True)
        dest = schema_dir / f"{spec.class_name}.json"
    else:
        raise ValueError("Provide schema_dir or output_path")

    dest.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return dest


def load_schema(name: str, schema_dir: Path) -> SchemaSpec:
    """Load a SchemaSpec by name from a directory.

    Args:
        name: Schema class name (without .json extension).
        schema_dir: Directory to search.

    Returns:
        The loaded SchemaSpec.

    Raises:
        FileNotFoundError: If the schema file doesn't exist.
    """
    path = schema_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    return SchemaSpec.model_validate_json(path.read_text(encoding="utf-8"))


def list_schemas(schema_dir: Path) -> list[SchemaSpec]:
    """List all saved schemas in a directory.

    Returns:
        List of SchemaSpec objects, sorted by class_name.
    """
    if not schema_dir.exists():
        return []
    specs = []
    for f in sorted(schema_dir.glob("*.json")):
        try:
            specs.append(SchemaSpec.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:
            continue  # skip malformed files
    return specs
```

Also add `from pathlib import Path` to the imports at the top of the file (after line 17).

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schema_gen.py::TestSchemaStorage -v`
Expected: 6 PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/schema_gen.py tests/test_schema_gen.py
git commit -m "feat: add schema save/load/list storage functions"
```

---

### Task 2: Add borderless table helper to cli_theme.py

**Files:**
- Modify: `mosaicx/cli_theme.py` (after `make_kv_table()`, line 134)
- Test: manual visual check (theme helper, no unit test needed)

**Step 1: Add make_clean_table helper**

Add to `mosaicx/cli_theme.py` after `make_kv_table()`:

```python
def make_clean_table(**kwargs: object) -> Table:
    """Create a borderless table with dim headers and clean spacing."""
    return Table(
        box=None,
        show_edge=False,
        pad_edge=False,
        header_style=MUTED,
        padding=(0, 2),
        **kwargs,
    )
```

**Step 2: Commit**

```bash
git add mosaicx/cli_theme.py
git commit -m "feat: add borderless table helper for clean list views"
```

---

### Task 3: Wire schema generate to auto-save + add --output flag

**Files:**
- Modify: `mosaicx/cli.py:362-384` (schema_generate command)
- Test: `tests/test_cli_integration.py`

**Step 1: Write failing test**

Add to `tests/test_cli_integration.py`:

```python
class TestSchemaManagement:
    """Tests for schema save/load/list/refine CLI commands."""

    def test_schema_generate_saves_file(self, runner: CliRunner, tmp_path: Path):
        """schema generate --output saves a JSON file."""
        out_file = tmp_path / "test_schema.json"
        result = runner.invoke(
            cli,
            [
                "schema", "generate",
                "--description", "Simple test with a name field",
                "--output", str(out_file),
            ],
        )
        # Without LLM this will fail, so just check the --output flag is accepted
        # The full E2E test requires an LLM
        assert "--output" not in result.output or result.exit_code in (0, 1)
```

Note: Full E2E testing of generate+save requires an LLM. The unit test validates the flag is wired.

**Step 2: Update schema_generate in cli.py**

Replace the `schema_generate` function (lines 362-384):

```python
@schema.command("generate")
@click.option("--description", type=str, required=True, help="Natural-language description of the schema.")
@click.option("--example-text", type=str, default="", help="Optional example document text for grounding.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Save schema to this path (default: ~/.mosaicx/schemas/).")
def schema_generate(description: str, example_text: str, output: Optional[Path]) -> None:
    """Generate a Pydantic schema from a description."""
    _configure_dspy()

    from .pipelines.schema_gen import SchemaGenerator, save_schema

    generator = SchemaGenerator()
    with theme.spinner("Generating schema... hold my beer", console):
        result = generator(description=description, example_text=example_text)

    # Save schema
    cfg = get_config()
    saved_path = save_schema(
        result.schema_spec,
        schema_dir=None if output else cfg.schema_dir,
        output_path=output,
    )

    console.print(theme.ok("Schema generated \u2014 it's alive!"))
    console.print(theme.info(f"Model: {result.compiled_model.__name__}"))
    console.print(theme.info(
        f"Fields: {', '.join(result.compiled_model.model_fields.keys())}"
    ))
    console.print(theme.info(f"Saved: {saved_path}"))

    theme.section("Schema Spec", console)
    from rich.json import JSON

    console.print(Padding(JSON(result.schema_spec.model_dump_json()), (0, 0, 0, 2)))
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_cli_integration.py -q`
Expected: all pass

**Step 4: Commit**

```bash
git add mosaicx/cli.py tests/test_cli_integration.py
git commit -m "feat: auto-save generated schemas with --output flag"
```

---

### Task 4: Implement schema list command

**Files:**
- Modify: `mosaicx/cli.py:387-390` (schema_list command)
- Test: `tests/test_cli_integration.py`

**Step 1: Write failing test**

Add to `TestSchemaManagement` in `tests/test_cli_integration.py`:

```python
    def test_schema_list_empty(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """schema list shows count when no schemas saved."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config
        get_config.cache_clear()
        try:
            result = runner.invoke(cli, ["schema", "list"])
            assert result.exit_code == 0
            assert "0 schema(s)" in result.output
        finally:
            get_config.cache_clear()

    def test_schema_list_shows_saved(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """schema list shows saved schemas in a table."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config
        get_config.cache_clear()
        try:
            # Create a schema file
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir()
            (schema_dir / "TestModel.json").write_text(
                '{"class_name":"TestModel","description":"A test","fields":[{"name":"x","type":"str","description":"x","required":true,"enum_values":null}]}'
            )
            result = runner.invoke(cli, ["schema", "list"])
            assert result.exit_code == 0
            assert "TestModel" in result.output
            assert "1 schema(s)" in result.output
        finally:
            get_config.cache_clear()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_integration.py::TestSchemaManagement -v`
Expected: FAIL

**Step 3: Implement schema_list in cli.py**

Replace the `schema_list` function (lines 387-390):

```python
@schema.command("list")
def schema_list() -> None:
    """List saved schemas."""
    from .pipelines.schema_gen import list_schemas

    cfg = get_config()
    specs = list_schemas(cfg.schema_dir)

    theme.section("Saved Schemas", console)

    if not specs:
        console.print(theme.info("0 schema(s) saved"))
        return

    t = theme.make_clean_table()
    t.add_column("Name", style=f"bold {theme.CORAL}", no_wrap=True)
    t.add_column("Fields", style="magenta", justify="right")
    t.add_column("Description")
    t.add_column("Path", style=theme.MUTED)

    for spec in specs:
        t.add_row(
            spec.class_name,
            str(len(spec.fields)),
            spec.description[:60] if spec.description else "\u2014",
            str(cfg.schema_dir / f"{spec.class_name}.json"),
        )

    console.print(t)
    console.print(theme.info(f"{len(specs)} schema(s) saved in {cfg.schema_dir}"))
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_cli_integration.py::TestSchemaManagement -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mosaicx/cli.py tests/test_cli_integration.py
git commit -m "feat: implement schema list with borderless table"
```

---

### Task 5: Add RefineSchemaSpec signature and SchemaRefiner module

**Files:**
- Modify: `mosaicx/pipelines/schema_gen.py` (inside `_build_dspy_classes()`, line 175)
- Test: `tests/test_schema_gen.py`

**Step 1: Write failing test**

Add to `tests/test_schema_gen.py`:

```python
class TestSchemaRefinerSignature:
    """Test the DSPy refinement signature exists and has correct fields."""

    def test_refine_signature_has_fields(self):
        from mosaicx.pipelines.schema_gen import RefineSchemaSpec

        sig = RefineSchemaSpec
        assert "current_schema" in sig.input_fields
        assert "instruction" in sig.input_fields
        assert "refined_schema" in sig.output_fields
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema_gen.py::TestSchemaRefinerSignature -v`
Expected: FAIL — `ImportError: cannot import name 'RefineSchemaSpec'`

**Step 3: Add RefineSchemaSpec and SchemaRefiner to _build_dspy_classes()**

Inside `_build_dspy_classes()` in `schema_gen.py`, after the `SchemaGenerator` class (before the return statement on line 226), add:

```python
    class RefineSchemaSpec(dspy.Signature):
        """Refine an existing schema based on user instructions.
        Preserve existing fields unless the instruction says to change them."""

        current_schema: str = dspy.InputField(
            desc="Current SchemaSpec as JSON"
        )
        instruction: str = dspy.InputField(
            desc="User instruction describing how to refine the schema"
        )
        refined_schema: SchemaSpec = dspy.OutputField(
            desc="The updated SchemaSpec incorporating the requested changes"
        )

    class SchemaRefiner(dspy.Module):
        """DSPy Module that refines an existing schema based on instructions."""

        def __init__(self) -> None:
            super().__init__()
            self.refine = dspy.ChainOfThought(RefineSchemaSpec)

        def forward(self, current_schema: str, instruction: str) -> dspy.Prediction:
            result = self.refine(
                current_schema=current_schema,
                instruction=instruction,
            )
            spec: SchemaSpec = result.refined_schema
            compiled = compile_schema(spec)
            return dspy.Prediction(
                schema_spec=spec,
                compiled_model=compiled,
            )
```

Update the return statement to include the new classes:

```python
    return GenerateSchemaSpec, SchemaGenerator, RefineSchemaSpec, SchemaRefiner
```

Update `_build_dspy_classes` cache and `__getattr__` to handle 4 classes:

```python
_DSPY_CLASS_NAMES = frozenset({
    "GenerateSchemaSpec", "SchemaGenerator",
    "RefineSchemaSpec", "SchemaRefiner",
})

def __getattr__(name: str):
    global _dspy_classes
    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            gen_sig, gen_mod, ref_sig, ref_mod = _build_dspy_classes()
            _dspy_classes = {
                "GenerateSchemaSpec": gen_sig,
                "SchemaGenerator": gen_mod,
                "RefineSchemaSpec": ref_sig,
                "SchemaRefiner": ref_mod,
            }
        return _dspy_classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_schema_gen.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/schema_gen.py tests/test_schema_gen.py
git commit -m "feat: add RefineSchemaSpec signature and SchemaRefiner module"
```

---

### Task 6: Add CLI-flag refinement helpers

**Files:**
- Modify: `mosaicx/pipelines/schema_gen.py` (after `list_schemas()`)
- Test: `tests/test_schema_gen.py`

**Step 1: Write failing tests**

Add to `tests/test_schema_gen.py`:

```python
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
```

**Step 2: Run test to verify they fail**

Run: `uv run pytest tests/test_schema_gen.py::TestSchemaFieldOps -v`
Expected: FAIL — `ImportError`

**Step 3: Implement field ops**

Add to `mosaicx/pipelines/schema_gen.py` after `list_schemas()`:

```python
# ---------------------------------------------------------------------------
# Field manipulation helpers (for CLI-flag refinement)
# ---------------------------------------------------------------------------

def add_field(spec: SchemaSpec, name: str, type_str: str, description: str = "", required: bool = True) -> SchemaSpec:
    """Return a new SchemaSpec with an additional field."""
    new_field = FieldSpec(name=name, type=type_str, description=description, required=required)
    return spec.model_copy(update={"fields": [*spec.fields, new_field]})


def remove_field(spec: SchemaSpec, name: str) -> SchemaSpec:
    """Return a new SchemaSpec with the named field removed."""
    new_fields = [f for f in spec.fields if f.name != name]
    if len(new_fields) == len(spec.fields):
        raise ValueError(f"Field {name!r} not found in schema")
    return spec.model_copy(update={"fields": new_fields})


def rename_field(spec: SchemaSpec, old_name: str, new_name: str) -> SchemaSpec:
    """Return a new SchemaSpec with a field renamed."""
    found = False
    new_fields = []
    for f in spec.fields:
        if f.name == old_name:
            new_fields.append(f.model_copy(update={"name": new_name}))
            found = True
        else:
            new_fields.append(f)
    if not found:
        raise ValueError(f"Field {old_name!r} not found in schema")
    return spec.model_copy(update={"fields": new_fields})
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_schema_gen.py::TestSchemaFieldOps -v`
Expected: 5 PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/schema_gen.py tests/test_schema_gen.py
git commit -m "feat: add field manipulation helpers for schema refinement"
```

---

### Task 7: Implement schema refine CLI command

**Files:**
- Modify: `mosaicx/cli.py:393-396` (schema_refine command)
- Test: `tests/test_cli_integration.py`

**Step 1: Write failing tests**

Add to `TestSchemaManagement` in `tests/test_cli_integration.py`:

```python
    def test_schema_refine_add_field(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """schema refine --add adds a field."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config
        get_config.cache_clear()
        try:
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir()
            (schema_dir / "TestModel.json").write_text(
                '{"class_name":"TestModel","description":"Test","fields":[{"name":"x","type":"str","description":"x","required":true,"enum_values":null}]}'
            )
            result = runner.invoke(
                cli, ["schema", "refine", "--schema", "TestModel", "--add", "y: int"]
            )
            assert result.exit_code == 0
            assert "y" in result.output
        finally:
            get_config.cache_clear()

    def test_schema_refine_remove_field(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """schema refine --remove removes a field."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config
        get_config.cache_clear()
        try:
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir()
            (schema_dir / "TestModel.json").write_text(
                '{"class_name":"TestModel","description":"Test","fields":[{"name":"x","type":"str","description":"x","required":true,"enum_values":null},{"name":"y","type":"int","description":"y","required":true,"enum_values":null}]}'
            )
            result = runner.invoke(
                cli, ["schema", "refine", "--schema", "TestModel", "--remove", "y"]
            )
            assert result.exit_code == 0
            assert "Removed" in result.output or "removed" in result.output
        finally:
            get_config.cache_clear()

    def test_schema_refine_rename_field(self, runner: CliRunner, tmp_path: Path, monkeypatch):
        """schema refine --rename renames a field."""
        monkeypatch.setenv("MOSAICX_HOME_DIR", str(tmp_path))
        from mosaicx.config import get_config
        get_config.cache_clear()
        try:
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir()
            (schema_dir / "TestModel.json").write_text(
                '{"class_name":"TestModel","description":"Test","fields":[{"name":"x","type":"str","description":"x","required":true,"enum_values":null}]}'
            )
            result = runner.invoke(
                cli, ["schema", "refine", "--schema", "TestModel", "--rename", "x=name"]
            )
            assert result.exit_code == 0
            assert "name" in result.output
        finally:
            get_config.cache_clear()
```

**Step 2: Run test to verify they fail**

Run: `uv run pytest tests/test_cli_integration.py::TestSchemaManagement -v`
Expected: FAIL

**Step 3: Implement schema_refine in cli.py**

Replace the `schema_refine` function (lines 393-396):

```python
@schema.command("refine")
@click.option("--schema", "schema_name", type=str, required=True, help="Name of the schema to refine.")
@click.option("--instruction", type=str, default=None, help="Natural-language refinement instruction (uses LLM).")
@click.option("--add", "add_field_str", type=str, default=None, help="Add a field: 'field_name: type'.")
@click.option("--remove", "remove_field_name", type=str, default=None, help="Remove a field by name.")
@click.option("--rename", "rename_str", type=str, default=None, help="Rename a field: 'old_name=new_name'.")
def schema_refine(
    schema_name: str,
    instruction: Optional[str],
    add_field_str: Optional[str],
    remove_field_name: Optional[str],
    rename_str: Optional[str],
) -> None:
    """Refine an existing schema."""
    from .pipelines.schema_gen import (
        load_schema, save_schema, compile_schema,
        add_field, remove_field, rename_field,
    )

    cfg = get_config()

    try:
        spec = load_schema(schema_name, cfg.schema_dir)
    except FileNotFoundError:
        raise click.ClickException(f"Schema not found: {schema_name}")

    old_field_names = {f.name for f in spec.fields}

    if instruction:
        # LLM-driven refinement
        _configure_dspy()
        from .pipelines.schema_gen import SchemaRefiner

        refiner = SchemaRefiner()
        with theme.spinner("Refining schema... one does not simply edit a schema", console):
            result = refiner(
                current_schema=spec.model_dump_json(indent=2),
                instruction=instruction,
            )
        spec = result.schema_spec

    elif add_field_str:
        # --add "field_name: type"
        parts = add_field_str.split(":", 1)
        if len(parts) != 2:
            raise click.ClickException("--add format: 'field_name: type'")
        fname, ftype = parts[0].strip(), parts[1].strip()
        spec = add_field(spec, fname, ftype)

    elif remove_field_name:
        try:
            spec = remove_field(spec, remove_field_name)
        except ValueError as exc:
            raise click.ClickException(str(exc))

    elif rename_str:
        parts = rename_str.split("=", 1)
        if len(parts) != 2:
            raise click.ClickException("--rename format: 'old_name=new_name'")
        old, new = parts[0].strip(), parts[1].strip()
        try:
            spec = rename_field(spec, old, new)
        except ValueError as exc:
            raise click.ClickException(str(exc))

    else:
        raise click.ClickException(
            "Provide --instruction, --add, --remove, or --rename"
        )

    # Verify the schema compiles
    try:
        compiled = compile_schema(spec)
    except Exception as exc:
        raise click.ClickException(f"Refined schema failed to compile: {exc}")

    # Save back
    save_schema(spec, schema_dir=cfg.schema_dir)

    # Show changes
    new_field_names = {f.name for f in spec.fields}
    added = new_field_names - old_field_names
    removed = old_field_names - new_field_names

    console.print(theme.ok("Schema refined \u2014 evolution, not revolution"))

    if added:
        for name in sorted(added):
            f = next(f for f in spec.fields if f.name == name)
            console.print(theme.info(f"+ {name} ({f.type})"))
    if removed:
        for name in sorted(removed):
            console.print(theme.info(f"- {name} (removed)"))

    # Show renamed (detected by same position, different name)
    if rename_str and "=" in rename_str:
        old_n, new_n = rename_str.split("=", 1)
        console.print(theme.info(f"~ {old_n.strip()} \u2192 {new_n.strip()}"))

    console.print(theme.info(f"Model: {compiled.__name__}"))
    console.print(theme.info(
        f"Fields: {', '.join(compiled.model_fields.keys())}"
    ))
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_cli_integration.py::TestSchemaManagement -v`
Expected: all PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/test_ocr_integration.py -q`
Expected: all pass

**Step 6: Commit**

```bash
git add mosaicx/cli.py tests/test_cli_integration.py
git commit -m "feat: implement schema refine with LLM and CLI-flag modes"
```

---

### Task 8: Final verification and E2E test

**Step 1: Run full test suite**

Run: `uv run pytest tests/ --ignore=tests/test_ocr_integration.py -q`
Expected: all pass

**Step 2: Manual E2E test with LLM**

```bash
# Generate and auto-save
mosaicx schema generate --description "Chest CT findings with nodule size, location, and density"

# List
mosaicx schema list

# Refine with CLI flag
mosaicx schema refine --schema ChestCTFindings --add "laterality: str"

# Verify update
mosaicx schema list
```

**Step 3: Commit any fixes**

If tests pass and E2E works, no additional commit needed.

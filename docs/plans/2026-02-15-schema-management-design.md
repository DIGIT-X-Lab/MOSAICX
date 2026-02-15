# Schema Management Design

## Problem

Generated schemas are ephemeral -- they print to stdout and disappear.
No way to save, list, or iteratively refine schemas.

## Goals

1. **Persist schemas** to disk after generation (auto-save)
2. **List saved schemas** with metadata
3. **Refine schemas** via LLM natural language or CLI flags

## Storage

- **Default location**: `~/.mosaicx/schemas/` (config.schema_dir)
- **Override**: `--output <path>` saves to a specific file
- **Format**: `SchemaSpec` JSON (`model_dump_json(indent=2)`)
- **Naming**: `{class_name}.json` (derived from LLM-generated class_name)
- **Loading**: `SchemaSpec.model_validate_json()` + `compile_schema()` to reconstruct Pydantic model

## CLI Commands

### `schema generate` (modify existing)

```bash
# Auto-saves to ~/.mosaicx/schemas/PatientRecord.json
mosaicx schema generate --description "Patient record with name and age"

# Save to specific path
mosaicx schema generate --description "..." --output ./my_schema.json
```

After generation, saves the SchemaSpec JSON to disk automatically.

### `schema list` (new)

```bash
mosaicx schema list
```

Displays a borderless table (dim headers, no box borders) with columns:
- **Name** (bold cyan)
- **Fields** (magenta, count)
- **Description** (default)
- **Path** (dim)

Footer: `info("N schema(s) saved")`

### `schema refine` (new)

Two modes:

**LLM mode** (`--instruction`):
```bash
mosaicx schema refine --schema PatientRecord \
  --instruction "add insurance ID and make age optional"
```

Pipeline:
1. Load existing SchemaSpec from disk
2. New DSPy signature `RefineSchemaSpec`: current_schema + instruction -> new SchemaSpec
3. Compile to verify validity
4. Show field-level diff (added/removed/changed)
5. Save back to same file

**CLI flags mode** (no LLM call):
```bash
mosaicx schema refine --schema PatientRecord --add "insurance_id: str"
mosaicx schema refine --schema PatientRecord --remove age
mosaicx schema refine --schema PatientRecord --rename "name=full_name"
```

Pipeline:
1. Load existing SchemaSpec from disk
2. Mutate fields list directly
3. Compile to verify validity
4. Show field-level diff
5. Save back

## DSPy Additions

### RefineSchemaSpec Signature

```python
class RefineSchemaSpec(dspy.Signature):
    """Refine an existing schema based on user instructions."""

    current_schema: str = dspy.InputField(desc="Current SchemaSpec as JSON")
    instruction: str = dspy.InputField(desc="User instruction for refinement")
    refined_schema: SchemaSpec = dspy.OutputField(desc="Updated SchemaSpec")
```

### SchemaRefiner Module

```python
class SchemaRefiner(dspy.Module):
    def __init__(self):
        self.refine = dspy.ChainOfThought(RefineSchemaSpec)

    def forward(self, current_schema: str, instruction: str):
        result = self.refine(current_schema=current_schema, instruction=instruction)
        compiled = compile_schema(result.refined_schema)
        return dspy.Prediction(
            schema_spec=result.refined_schema,
            compiled_model=compiled,
        )
```

## UI Style

- Borderless tables with dim headers, clean column spacing
- Uses existing `theme.section()`, `theme.ok()`, `theme.info()` for consistency
- Meme completion messages:
  - generate: "Schema generated -- it's alive!" (already done)
  - list: info line with count
  - refine: "Schema refined -- evolution, not revolution"

## Files to Modify

| File | Change |
|------|--------|
| `mosaicx/pipelines/schema_gen.py` | Add `save_schema()`, `load_schema()`, `list_schemas()`, `RefineSchemaSpec`, `SchemaRefiner` |
| `mosaicx/cli.py` | Wire `schema list`, `schema refine`, add `--output` to `schema generate`, auto-save |
| `mosaicx/cli_theme.py` | Add `make_borderless_table()` helper |
| `tests/test_schema_gen.py` | Tests for save/load/list/refine |

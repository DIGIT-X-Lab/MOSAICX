# BUG: Nested object fields lost in template generation

**Status**: Fixed (2026-02-23)
**Discovered**: 2026-02-23
**Severity**: High — blocks extraction for any template with `list[object]` sections
**Affects**: `mosaicx template create --from-radreport`, `mosaicx template create --describe`

## Symptom

```bash
mosaicx extract --document report.pdf --template MRICervicalSpineReport
# Error: Object field 'None' must provide 'fields'.
```

Any template containing a `list` of `object` items crashes at extraction time.

## Reproduction

```bash
mosaicx template create --from-radreport RPT50890 --name MRICervicalReportV2
mosaicx extract --document /path/to/cervical_spine.pdf --template MRICervicalReportV2
```

The generated template YAML contains:

```yaml
- name: level_findings
  type: list
  item:
    type: object          # <-- no 'fields' key
  description: '...'      # <-- nested field info buried here as text
```

The template compiler (`template_compiler.py:165-169`) requires `fields` when `type: object`:

```python
if type_str == "object":
    if not spec.fields:
        raise ValueError(
            f"Object field '{getattr(spec, 'name', '?')}' must provide 'fields'."
        )
```

## Root Cause

Two incompatible `FieldSpec` classes exist:

| Model | File | Has `fields`? |
|-------|------|---------------|
| Pipeline `FieldSpec` | `mosaicx/pipelines/schema_gen.py:31-48` | No |
| Template `FieldSpec` | `mosaicx/schemas/template_compiler.py:38-51` | Yes |

**The pipeline's `FieldSpec` has no `fields` attribute.** When the CLI instructs the LLM to use `list[object]` for repeating structured data, the LLM cannot express nested field definitions. It puts them in the `description` string instead. The YAML converter (`schema_spec_to_template_yaml()`) then emits `item: {type: object}` with no `fields` key, producing invalid YAML.

### Execution path

1. **CLI** (`cli.py:1161-1166`) tells LLM to use `list[object]` for repeating data
2. **LLM** generates `SchemaSpec` with `FieldSpec(type="list[object]", description="...fields as text...")`
3. **`schema_spec_to_template_yaml()`** (`template_compiler.py:291`) creates `section["item"] = {"type": "object"}` — no fields
4. **YAML saved** with broken structure
5. **`compile_template()`** reads YAML, hits `_resolve_type()`, crashes at line 166

### Key code location

```python
# template_compiler.py:291 — the line that drops nested fields
section["item"] = {"type": inner}  # {"type": "object"} only, no fields!
```

## Fix

### 1. Add `fields` to pipeline FieldSpec (`schema_gen.py:31-48`)

```python
fields: Optional[List["FieldSpec"]] = Field(
    None,
    description="Sub-fields when type is 'object' or 'list[object]'",
)
```

Backward compatible — `Optional`, defaults to `None`. Existing saved SchemaSpec JSON files deserialize without error.

### 2. Handle nested objects in `_resolve_type` (`schema_gen.py:99-109`)

In the `list[X]` branch, when `inner == "object"` and `spec.fields` exists, build a nested Pydantic model via `create_model()` instead of `list[dict]`. When `fields` is `None`, fall back to `list[dict]`.

### 3. Emit nested fields in YAML converter (`template_compiler.py:280-291`)

When `inner == "object"` and `field.fields` exists:

```python
section["item"] = {
    "type": "object",
    "fields": [_field_spec_to_dict(sub) for sub in field.fields],
}
```

Add `_field_spec_to_dict()` helper for recursive serialization.

**Fallback**: When `inner == "object"` but `field.fields` is `None`, downgrade to `item: {type: str}` instead of emitting broken YAML.

### 4. Update LLM signature docstring (`schema_gen.py:401-413`)

Add guidance for `list[object]` with `fields` attribute.

## Files to modify

- `mosaicx/pipelines/schema_gen.py` — FieldSpec model, `_resolve_type`, LLM signature
- `mosaicx/schemas/template_compiler.py` — `schema_spec_to_template_yaml()`
- `tests/test_schema_gen.py` — nested FieldSpec construction and compilation tests
- `tests/test_report.py` — roundtrip YAML generation/compilation tests

## Verification

1. `pytest tests/test_schema_gen.py tests/test_report.py -q` — all tests pass
2. `make lint` — no violations
3. Re-run `mosaicx template create --from-radreport RPT50890` and confirm the YAML contains proper `fields` under `item: type: object`
4. Extract with the regenerated template — no crash

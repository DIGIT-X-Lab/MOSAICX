# Run Log: BUG-001 Nested Object Template Fix

Date: 2026-02-23
Area: template/schema generation
Status: pass

## 1) Environment
- Repo: `MOSAICX`
- Python: `.venv/bin/python`
- Test runner: `pytest`

## 2) Commands
- `sed -n '1,240p' docs/bugs/nested-object-fields-lost-in-template-generation.md`
- `.venv/bin/pytest -q tests/test_schema_gen.py tests/test_report.py`

## 3) Observed outputs
- Root cause validated: pipeline `FieldSpec` could not represent nested object fields, and YAML conversion emitted invalid `item: {type: object}` without `fields`.
- After fix, nested list-object fields are preserved in generated YAML and compile correctly.
- Test summary: `68 passed`.

## 4) Failures
- None in targeted regression suite.

## 5) Follow-up fixes
- Remaining operational check: run `mosaicx template create --from-radreport ...` + `mosaicx extract --template ...` with real report ID and source file to confirm end-to-end CLI behavior in integration.

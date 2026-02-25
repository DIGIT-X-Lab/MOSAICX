# EXTX-06 Implementation Artifact

Date: 2026-02-25
Issue: #64 (`EXTX-06`)
Branch: `feat/dspy-rlm-evolution`

## Scope implemented
- Added deterministic semantic validators in `mosaicx/pipelines/extraction.py` for:
  - null semantics (`missing_value`, `nullish_string`)
  - date normalization/validation (to `YYYY-MM-DD`)
  - numeric range validation (`invalid_range_format`, `invalid_range_order`)
  - numeric+unit validation (`invalid_numeric_unit`, `unknown_unit`)
  - numeric-content checks for numeric-like fields (`missing_numeric_content`)
- Added blood-pressure aware normalization for values like `128/82`.
- Extended extraction contract output:
  - per-field `validation` block (`valid`, `kind`, `reason`, `critical`)
  - top-level `_extraction_contract.validation_issues` list
- Contract now deterministically downgrades invalid unsupported values to
  `needs_review` or `insufficient_evidence` (critical fields).

## Regression test additions
- `tests/test_extract_contract.py`
  - date normalization + invalid range downgrade coverage
  - unknown unit downgrade coverage

## Notes
- This artifact captures implementation progress; execution of test matrix is deferred.

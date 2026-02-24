## Problem
`mosaicx verify --level thorough --claim ...` currently returns `effective_level=deterministic` due to claim-path short-circuiting in verify engine.

## Why this matters
- Breaks trust in level semantics.
- Produces fallback flags even when local LLM is healthy.
- Prevents true RLM audit behavior for claim checks.

## Repro
`PYTHONPATH=. .venv/bin/mosaicx verify --sources tests/datasets/extract/sample_patient_vitals.pdf --claim "patient BP is 128/82" --level thorough -o /tmp/out.json`

## Expected
- `requested_level=thorough`
- `effective_level=audit`
- `fallback_used=false` unless audit truly fails.

## Acceptance Criteria
- Claim path supports `quick/standard/thorough` properly.
- Thorough claim checks invoke audit/RLM path.
- SDK/CLI integration tests assert level contract for claim mode.
- Fallback reason is emitted only on real degradation events.

## References
- `mosaicx/verify/engine.py`
- `mosaicx/sdk.py`

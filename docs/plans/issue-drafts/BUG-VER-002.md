## Problem
Verify output can present a top-line message consistent with full audit while JSON fields indicate deterministic fallback.

## Why this matters
- Confusing for developers integrating SDK/CLI.
- Hard to know which signal is canonical.

## Scope
Unify and enforce one truth contract for verify outputs (CLI + JSON).

## Acceptance Criteria
- Canonical truth fields are explicitly documented and stable (`decision`, `claim_true`, `effective_level`, `fallback_used`).
- CLI rendering reflects actual `effective_level`.
- Contradiction/support blocks are consistent with machine payload.
- Regression tests cover supported and contradicted claims.

## References
- existing verify contract work (#21)
- `mosaicx/cli.py`
- `mosaicx/sdk.py`

## Problem
Fallback/rescue layers can hide primary-path regressions (e.g., deterministic rescue producing an answer after planner/audit failed), making failures less visible.

## Scope
Add quality gates that assert primary-route behavior for key scenarios before fallback is allowed.

## Acceptance Criteria
- Tests record primary route vs fallback/rescue route.
- Key flows (query analytics, verify thorough claim) pass on primary route.
- Fallback remains available but is explicitly reported as degraded mode.

## References
- evaluation roadmap issues (#46, #48, #49, #50)
- `tests/test_query_engine.py`
- `tests/test_sdk_verify.py`

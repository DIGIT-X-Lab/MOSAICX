## Problem
In chat/EDA flows, follow-up prompts can drift to wrong columns/entities (e.g., answering Weight when asked for number of columns, or Subject when asked for sex/gender distribution).

## Why this matters
- Produces incorrect analytics despite deterministic tools being available.
- Reduces trust in agentic query behavior.

## Scope
Strengthen planner-state-executor coupling and answer verification for follow-up questions.

## Acceptance Criteria
- Follow-up/coreference prompts preserve prior entity/metric intent.
- Wrong-column execution is detected and blocked before narration.
- Integration tests include typo/coreference/ambiguous prompts.
- Evidence explicitly supports returned metric/value.

## References
- roadmap query hardening (#34)
- DSPY planner issue (#36)
- `mosaicx/query/engine.py`
- `mosaicx/query/control_plane.py`

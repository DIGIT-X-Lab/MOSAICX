## Problem
LLM-dependent tests/runs can be started without strict preflight, leading to false negatives, flaky behavior, or misleading results.

## Required guards
- Verify local vLLM endpoint is reachable and serving expected model.
- Verify one healthy chat-completion round-trip before integration suite.
- Enforce cache cleanup policy where intended by test profile.

## Acceptance Criteria
- Hard-test runner fails fast if LLM preflight fails.
- Test report captures model id, endpoint, and preflight timestamp.
- Cache policy (clear/reuse) is explicit per profile and logged.

## References
- ops roadmap issue (#33)
- `scripts/run_hard_test_matrix.sh`

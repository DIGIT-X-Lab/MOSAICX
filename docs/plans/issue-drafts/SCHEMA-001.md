Problem
- Template/schema generation is stable on happy paths but can degrade on noisy inputs and ambiguous prompts.
- Generated SchemaSpec can contain weak field naming/type choices that still pass compile but reduce extraction quality.
- Current generator relies mostly on a single ChainOfThought pass.

Scope
- Harden schema generation with explicit normalize/validate/repair stages.
- Add DSPy candidate selection (BestOfN when available) plus deterministic quality gates.
- Ensure output SchemaSpec and input document context are both validated before save/compile.

Acceptance Criteria
- Schema generation applies deterministic normalization for class/field identifiers and type aliases.
- Validation catches/repairs duplicate/invalid names, empty enums, and structurally invalid nested object/list fields.
- Generator attempts repair when validation/compile fails, then revalidates.
- BestOfN selection is used when DSPy supports it; fallback remains safe when unavailable.
- CLI `template create` fails with clear actionable errors if robust generation cannot produce a valid schema.
- Regression tests added for noisy/edge prompts and malformed intermediate specs.
- Live checks: `template create --describe` and `--from-document` produce valid templates under local vLLM-MLX.

References
- docs/plans/2026-02-24-roadmap-status-audit.md
- docs/plans/2026-02-23-production-rlm-roadmap.md

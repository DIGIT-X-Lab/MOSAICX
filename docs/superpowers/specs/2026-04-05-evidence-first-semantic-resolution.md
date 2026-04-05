# Evidence-First Semantic Resolution

**Date:** 2026-04-05  
**Status:** Proposal  
**Audience:** extraction architecture, provenance, SDK design

## Summary

MOSAICX currently mixes three different ideas:

1. extracted value
2. source grounding
3. normalization / canonicalization

That produces duplicated logic, inconsistent evidence, and brittle behavior on semantically correct values whose normalized form does not literally appear in the source document.

Example:

- source text: `62Y/F`
- extracted value: `Female`
- current deterministic grounding may fail or ground to an unrelated occurrence of `Female`

This proposal replaces the current "search the final value again" approach with an **Evidence-First Semantic Resolution** layer:

1. extract a candidate value
2. anchor exact source evidence
3. canonicalize from the anchored evidence
4. emit both the raw source value and the normalized SDK value

The design goal is developer-grade structured extraction that is:

- semantically robust
- auditable
- low-latency
- easier to reason about than the current layered provenance/contract duplication

## Problem

Today there are effectively three overlapping grounding systems:

1. **Schema augmentation**
   - the extraction model can emit `{field}_excerpt` and `{field}_reasoning`
   - rich, but self-reported by the same model that extracted the value

2. **Source mapping**
   - deterministic text search and coordinate lookup produce `_source`
   - this is the strongest current trust boundary because it is deterministic and coordinate-aware

3. **Extraction contract**
   - runs another fuzzy grounding pass via `_extract_grounding_snippet()`
   - semantically overlaps with `_source` but is weaker and coordinate-free

This causes several concrete problems:

- duplicated text search over the same document
- inconsistent "grounded" judgments between `_source` and `_extraction_contract`
- false ungrounded flags when the normalized output differs from the raw source token
- no explicit representation of normalization
- poor handling of shorthand, abbreviations, enums, and registrar-style aliases

## Design Principle

**No semantic normalization without grounded evidence.**

The system should not ask:

- "Does the final normalized value appear verbatim in the document?"

It should ask:

1. "What exact source evidence supports this field?"
2. "Given that evidence, what is the canonical output value?"

That distinction is the core architectural change.

## Proposed Architecture

```text
Document
  -> OCR / text loading
  -> extraction
  -> field evidence anchoring
  -> semantic canonicalization
  -> deterministic validation
  -> canonical _source
  -> derived extraction contract
```

### Step 1: Extraction

Keep the current main extractor behavior:

- Outlines or other structured path returns a field candidate
- optional model-supplied `_excerpt` and `_reasoning` are still allowed

Important:

- model evidence is useful, but it is not the source of truth

### Step 2: Evidence Anchoring

Create one canonical grounding layer, backed by `_source`.

Anchoring order:

1. use model-supplied excerpt if present
2. locate the excerpt deterministically in `doc.text`
3. resolve spans / coordinates from `LoadedDocument.text_blocks`
4. if excerpt is missing, fall back to locating the raw field candidate

Anchoring output:

- raw excerpt
- raw token or source value when recoverable
- char offsets
- page and bbox spans
- anchor confidence
- anchor method

### Step 3: Semantic Canonicalization

Run a bounded resolver on the anchored evidence, not on the whole document.

Inputs:

- field name
- raw source excerpt
- raw source token if available
- candidate extracted value
- schema field type
- allowed values or ontology, when available

Outputs:

- canonical SDK value
- normalization metadata
- confidence

The canonicalizer is a resolver stack:

1. deterministic parser
2. deterministic alias / enum resolver
3. ontology matcher
4. bounded semantic micro-model
5. passthrough

The first successful high-confidence resolver wins.

### Step 4: Deterministic Validation

Validation still runs after canonicalization.

Use existing validators for:

- dates
- units
- numeric ranges
- numeric coercion
- blood pressure
- null / requiredness

But validation should now operate on:

- canonical output value

while preserving:

- raw source evidence
- normalization metadata

### Step 5: Canonical `_source`

`_source` becomes the single authoritative provenance object.

Proposed per-field structure:

```json
{
  "value": "Female",
  "source_value": "F",
  "excerpt": "Mrs SAKUNTHALA ... 62Y/F ...",
  "grounded": true,
  "spans": [
    {"page": 1, "bbox": [43.5, 710.5, 197.0, 733.0]}
  ],
  "anchor": {
    "method": "llm_excerpt_then_locate",
    "confidence": 0.97
  },
  "canonicalization": {
    "applied": true,
    "method": "semantic_enum",
    "from": "F",
    "to": "Female",
    "confidence": 0.99
  },
  "reasoning": "The source token 'F' denotes female sex.",
  "llm_excerpt": "62Y/F"
}
```

This separates:

- what the document said
- what the SDK returns
- how the mapping happened

### Step 6: Derived `_extraction_contract`

`_extraction_contract` should no longer run an independent text search.

Instead it should derive its fields from:

- `_source.fields[field]`
- deterministic semantic validation
- required-field rules

That makes the contract a summary view, not a third provenance engine.

## Resolver Stack

### 1. Deterministic Parser

Use when a field has stable formatting:

- dates
- timestamps
- blood pressure
- units
- numeric ranges
- percentages

Examples:

- `28-03-2026 12:27:17` -> `2026-03-28`
- `120 / 82` -> `120/82`
- `4.2 g/dl` -> `4.2 g/dl`

### 2. Deterministic Alias / Enum Resolver

Use when mapping is closed-world and high-confidence:

- `F` -> `Female`
- `M` -> `Male`
- `R` -> `Right`
- `L` -> `Left`
- `Pos` -> `Positive`
- `Neg` -> `Negative`

This layer should be schema-aware, not a giant global alias table.

### 3. Ontology Matcher

Use for registrar-style fields or other controlled vocabularies:

- specimen type
- laterality
- tumor behavior
- morphology labels
- body site
- staging categories

It can combine:

- alias tables
- token overlap
- ontology synonyms
- deterministic scores

### 4. Bounded Semantic Micro-Model

Only run when the earlier layers cannot confidently resolve a field.

Important constraints:

- input is the anchored snippet, not the full document
- output must be structured
- only invoked for a small set of ambiguous fields
- ideally batched across multiple fields in one tiny call

Example input:

```json
{
  "fields": [
    {
      "field": "sex",
      "source_value": "F",
      "excerpt": "Mrs SAKUNTHALA 62Y/F",
      "allowed_values": ["Male", "Female", "Other", "Unknown"]
    },
    {
      "field": "sample_type",
      "source_value": "EDTA BLOOD",
      "excerpt": "Sample Type: EDTA BLOOD",
      "allowed_values": ["SERUM", "PLASMA", "EDTA BLOOD", "WHOLE BLOOD"]
    }
  ]
}
```

Example output:

```json
{
  "fields": [
    {
      "field": "sex",
      "value": "Female",
      "method": "semantic_enum",
      "confidence": 0.99
    },
    {
      "field": "sample_type",
      "value": "EDTA BLOOD",
      "method": "enum_identity",
      "confidence": 1.0
    }
  ]
}
```

### 5. Passthrough

Use when no canonicalization is needed:

- free text fields
- already grounded exact values
- unsupported field classes

## Latency Model

The semantic layer must not become another orchestration tax.

### Hot-path rule

Most documents should incur:

- 1 extraction call
- 0 semantic calls

### When semantic resolution runs

Only run semantic resolution if all are true:

1. field is configured as semantically normalizable
2. field is grounded or partially grounded
3. deterministic resolver confidence is below threshold
4. the field materially benefits from canonicalization

### Batching

Semantic micro-resolution should batch all ambiguous enum-like fields into one small structured call.

This keeps cost much lower than:

- planner LLM routing
- BestOfN
- adjudication
- deep repair/refine loops

### Expected latency profile

- common fast extraction: unchanged or near-unchanged
- ambiguous enum-heavy docs: +1 tiny structured call
- far cheaper than current multi-layer rescue architecture

## Proposed Data Model

Introduce a field-resolution object internally:

```python
@dataclass
class FieldResolution:
    field: str
    candidate_value: Any
    value: Any
    source_value: str | None
    excerpt: str | None
    grounded: bool
    spans: list[dict[str, Any]]
    anchor_method: str | None
    anchor_confidence: float | None
    canonicalization_applied: bool
    canonicalization_method: str | None
    canonicalization_confidence: float | None
    reasoning: str | None
    llm_excerpt: str | None
```

This becomes the internal handoff object between:

- extraction
- provenance
- normalization
- contract generation

## Migration Plan

### Phase 1: Unify current evidence flow

Goal:

- make `_source` the canonical grounding layer without changing public behavior much

Changes:

- pass `field_evidence` directly into `build_source_block()` on the main CLI and SDK paths
- remove the duplicate post-build evidence merge in the CLI
- stop treating `_extraction_contract` as an independent grounding engine
- derive contract grounding from `_source`

### Phase 2: Introduce explicit normalization metadata

Goal:

- distinguish raw source evidence from final normalized value

Changes:

- add `source_value`
- add `canonicalization`
- preserve the current `value`

### Phase 3: Add resolver stack

Goal:

- move normalization from implicit LLM behavior to explicit resolution

Changes:

- deterministic parser module
- enum / alias resolver
- ontology matcher
- bounded semantic micro-model fallback

### Phase 4: Registrar-grade ontology support

Goal:

- support high-value controlled vocab fields with explicit semantics

Examples:

- laterality
- specimen type
- site
- histology
- behavior
- stage families

## Concrete Changes to Current Code

### Keep

- `_augment_schema_with_evidence()` in `mosaicx/pipelines/extraction.py`
- `build_source_block()` in `mosaicx/source_mapping.py`
- deterministic semantic validators in `mosaicx/pipelines/extraction.py`

### Change

- `build_source_block()` should become the canonical place where field evidence and grounding are unified
- `apply_extraction_contract()` should consume `_source`, not call `_extract_grounding_snippet()` independently
- CLI and SDK should pass `field_evidence` directly to `_source` construction on the main path

### Add

- a new semantic resolver module, for example:
  - `mosaicx/pipelines/semantic_resolution.py`
- a field-resolution internal representation
- config or schema hooks that mark which fields are eligible for semantic normalization

### Remove or de-emphasize

- duplicated fuzzy grounding in `_extraction_contract`
- the assumption that grounded means "normalized final value appears literally in source text"

## Example

### Input

Source excerpt:

```text
Mrs SAKUNTHALA 62Y/F
```

Extracted candidate:

```json
{
  "sex": "Female"
}
```

### Current failure mode

- search for `Female`
- fail, or match unrelated `Female` elsewhere
- mark field as weakly grounded or inconsistently grounded

### Proposed behavior

1. anchor excerpt `62Y/F`
2. recover `source_value = "F"`
3. canonicalize `F -> Female`
4. emit:

```json
{
  "value": "Female",
  "source_value": "F",
  "grounded": true,
  "canonicalization": {
    "applied": true,
    "method": "semantic_enum",
    "from": "F",
    "to": "Female",
    "confidence": 0.99
  }
}
```

## Why this is better

- semantically correct values stop looking ungrounded
- provenance becomes auditable
- normalization becomes explicit
- developer UX improves because the system explains both evidence and canonical value
- latency stays bounded if semantic resolution is sparse and batched
- the architecture is simpler than maintaining multiple independent evidence systems

## Open Questions

These are the questions worth asking in review:

1. Should `_source` fully replace `_extraction_contract` as the source of grounding truth, or should the contract keep a reduced summary-only role?
2. Should semantic resolution be schema-declared per field, or inferred from field names and types?
3. For closed-world fields, should the bounded semantic resolver always receive allowed values, or should ontology lookup happen before model invocation?
4. Should registrar-specific ontologies live in template metadata, code, or a separate normalization registry?
5. What confidence threshold should gate a semantic micro-model result before falling back to `needs_review`?

## Recommendation

Adopt this design incrementally.

The first implementation step is not the semantic micro-model. It is:

1. unify `_source` as the canonical grounding layer
2. remove duplicate grounding work from `_extraction_contract`
3. add explicit raw-vs-normalized provenance fields

After that, add bounded semantic resolution only for the field classes that genuinely need it.

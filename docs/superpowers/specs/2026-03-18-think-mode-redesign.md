# Think Mode Redesign — Proportional Accuracy Extraction

> **For agentic workers:** Use superpowers:executing-plans to implement this spec.

## Problem

The current `--think` modes (fast/standard/deep) don't deliver meaningful accuracy differences:
- `fast` and `standard` produce similar results
- `deep` runs two extraction paths that converge on identical output
- No deterministic compute on derived fields (LLM hallucinates arithmetic)
- No self-verification (errors go undetected)

## Design

### User Interface

```bash
mosaicx extract --document report.pdf --template ProstataPatient           # auto (default)
mosaicx extract --document report.pdf --template CTLiverLesionClassifier --think fast
mosaicx extract --document report.pdf --template ProstataPatient --think deep
```

Three flags: `auto` (default), `fast`, `deep`. Standard exists internally but isn't exposed — the router handles it.

### Router (Deterministic, No LLM)

Analyzes the template YAML to pick the think level:

```
fast     ← fields <= 8, no list[object] nesting
standard ← fields <= 20, nesting_depth <= 2, no calculated fields
deep     ← everything else (nested lists, calculated fields, 20+ fields)
```

Detection rules:
- **Field count**: `len(sections)` + recursive count of nested fields
- **Nesting depth**: walk the template tree, count max depth of `list[object]`
- **Calculated fields**: scan field descriptions for keywords: "count", "calculate", "max(", "sum(", "number of", "percentage"

Override behavior:
- `--think auto`: router picks freely (default)
- `--think fast`: `max(fast, template_floor)` — won't produce garbage on complex schemas
- `--think deep`: always deep, user wants maximum accuracy

### Pipeline: Fast

One LLM call. No reasoning.

```
Document + Schema → Outlines (constrained JSON) → Compute → Result
```

- Outlines constrains output to valid JSON matching the schema
- Fallback: `dspy.Predict` if Outlines fails
- Compute: deterministic overwrite of calculated fields

### Pipeline: Standard

One LLM call with step-by-step reasoning.

```
Document + Schema → ChainOfThought → Compute → Consistency Checks → Result
```

- CoT reasons through each field before producing output
- Compute: deterministic overwrite of calculated fields
- Consistency checks (deterministic):
  - List counts match `anzahl_*` fields
  - Enum values are valid
  - Dates parse correctly
  - Required fields are non-null

### Pipeline: Deep

Multi-pass extraction with chunking, self-verification, and targeted repair.

```
PASS 1: CHUNKED EXTRACT
  - Walk schema tree, split at each list[object] boundary
  - Chunk 1: all scalar/enum fields → one CoT call
  - Chunk 2: first list[object] → one CoT call
  - Chunk N: nth list[object] → one CoT call
  - Assemble chunks into complete extraction

COMPUTE
  - Deterministic overwrite of all calculated fields

PASS 2: VERIFY
  - Single LLM call with: original document + assembled extraction
  - Prompt: "Review this extraction against the source document.
    For each field, is the value correct? Flag anything wrong or missing."
  - Output: list of {field, issue, evidence_from_document}

PASS 3: FIX
  - For each flagged field: targeted CoT call with
    the specific error + relevant document section
  - Recompute derived fields after fixes
```

### Chunking Logic

Walk the template YAML tree. Every `list` with `item.type: object` is a chunk boundary. All other fields (scalars, enums, simple lists) group into one "scalar chunk."

Example for ProstataPatient:
- **Chunk 1 (scalars)**: patientenid, dates, institut, enums, classifications
- **Chunk 2 (makroskopie_liste)**: list of {nr, gesamt_stanzen, laenge_cm}
- **Chunk 3 (begutachtung_liste)**: list of {nr, tumor, tumorausdehnung_mm, gleason}

Each chunk gets its own CoT call with only the relevant document section as context. The chunking groups fields from the same document section together, reducing attention dilution.

### Deterministic Compute

Runs at ALL think levels. Detects calculated fields from template descriptions and overwrites LLM values with deterministic results.

Detection: scan field descriptions for patterns:
- "count all entries in X" → `len(extracted.X)`
- "count entries where Y is true" → `sum(1 for item in X if item.Y)`
- "calculate: max(A / B * 100)" → `max(a/b*100 for a,b in zip(...))`
- "number of" + references another list field → `len(that_list)`
- "percentage" + references two numeric fields → `a / b * 100`

Fallback: if description doesn't match any pattern, keep the LLM value.

### Verification Prompt (Deep Mode)

```
You are reviewing a structured extraction for correctness.

ORIGINAL DOCUMENT:
{document_text}

EXTRACTED DATA:
{extraction_json}

SCHEMA DESCRIPTION:
{template_description}

Review each extracted field against the original document.
For each field that is WRONG or MISSING, output:
- field: the field name (dot-notation for nested, e.g. "befunde.0.histologiedatum")
- issue: what's wrong (e.g. "Used Ausgang date instead of Eingang date")
- evidence: the relevant text from the document that shows the correct value
- suggested_value: the correct value

Output ONLY fields with issues. If everything is correct, output an empty list.
```

### Fix Prompt (Deep Mode, per flagged field)

```
Extract the correct value for the field "{field_name}".

FIELD DESCRIPTION: {field_description}
ISSUE FOUND: {issue}
EVIDENCE FROM DOCUMENT: {evidence}
RELEVANT DOCUMENT SECTION: {section_text}

Return ONLY the correct value for this field.
```

## Architecture

### Files to modify

- `mosaicx/pipelines/extraction.py` — core changes:
  - New: `_route_think_level(template_spec)` — deterministic router
  - New: `_chunk_schema(template_spec)` — split schema at list[object] boundaries
  - New: `_compute_derived_fields(extracted, template_spec, source_text)` — deterministic compute
  - New: `_verify_extraction(extracted, document_text, template_spec)` — verification pass
  - New: `_fix_flagged_fields(flagged, document_text, template_spec)` — targeted repair
  - Modified: `_extract_schema_with_structured_chain()` — integrate new pipeline
  - Modified: `DocumentExtractor.forward()` — wire router + new passes

- `mosaicx/config.py` — change `--think` default from `"standard"` to `"auto"`

- `mosaicx/cli.py` — update `--think` choices to `["auto", "fast", "deep"]`

### What NOT to change

- Outlines integration (used by fast mode)
- ChainOfThought extraction signatures (used by standard + deep)
- Planner / ReAct routing for long documents (orthogonal concern)
- Template YAML format (read-only for router)
- Scoring functions (kept for diagnostics, not used for routing)

## Testing

Accuracy test matrix — run all three modes on:
1. CTLiverLesionClassifier + text report (simple, flat)
2. chest_ct + PDF report (moderate, standard radiology)
3. ProstataPatient + 5 JPG pages (complex, nested, calculated)

Measure per-field accuracy against manually labeled ground truth.

Expected results:
- Fast: ~70% on ProstataPatient, ~95% on CTLiverLesionClassifier
- Standard: ~85% on ProstataPatient, ~95% on CTLiverLesionClassifier
- Deep: ~95% on ProstataPatient, ~95% on CTLiverLesionClassifier
- Auto picks the right level for each template

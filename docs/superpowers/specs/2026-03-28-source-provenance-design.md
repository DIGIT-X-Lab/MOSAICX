# Source Provenance & Coordinate Mapping

**Date:** 2026-03-28
**Branch:** `feat/source-provenance`
**Status:** Design (v2 — post-review)

## Problem

MOSAICX extracts structured data from medical documents and deidentifies PHI, but provides no mapping back to the source document. A GUI needs to know *where* on a PDF page each extracted field or redacted PHI was found — page number, bounding box coordinates — to render highlights.

Currently:
- pypdfium2 has per-character bounding boxes — discarded during text extraction.
- PaddleOCR has polygon coordinates per text block — discarded during OCR.
- The deidentifier's redaction map has character offsets but no page/bbox coordinates.
- The extraction pipelines provide structured output with no source location data at all.

## Solution

Three layers:

1. **Document layer** — capture coordinate data during text extraction, store as `TextBlock` objects on `LoadedDocument`.
2. **Pipeline layer** — a dedicated post-extraction evidence step produces an evidence map (field name → source excerpt). Deidentifier enriches its redaction map with coordinates.
3. **Resolution layer** — a shared utility matches excerpts to source text positions, then looks up coordinates from the TextBlock map.

Provenance is **opt-in** via `--provenance` flag (CLI) / `provenance=True` (SDK/MCP). The evidence step adds one LLM call, so users who do not need highlighting avoid the overhead.

## Critical Invariant

**The text passed to any pipeline MUST be exactly `doc.text` from the `LoadedDocument`, unmodified.** The TextBlock offsets reference character positions in `doc.text`. Any intermediate text processing (whitespace normalization, truncation, encoding changes) between document loading and pipeline invocation will break all provenance. This invariant must be tested.

## Layer 1: Document — TextBlock Map

### TextBlock Model

```python
@dataclass
class TextBlock:
    """A contiguous text region with its location on the source page."""
    text: str
    start: int    # char offset in full document text
    end: int      # char offset in full document text
    page: int     # 1-indexed page number
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in page points
```

`TextBlock` is internal — never serialized to output JSON. Only resolved provenance records appear in output.

### Per-Backend Construction

**pypdfium2 (native PDF text):**
- Iterate characters via `textpage.get_charbox(i)`.
- Group adjacent characters into word-level TextBlocks (split on whitespace/newline).
- One TextBlock per word — compact and precise enough for highlighting.
- Performance note: `get_charbox()` iterates every character. For a typical 3-page report (~5000 chars), this is fast (<50ms). For very large documents (50+ pages), consider page-level batching or lazy construction.

**PaddleOCR (scanned documents):**
- Extract `dt_polys` from `res` dict alongside `rec_texts`.
- One TextBlock per `rec_text` entry. Bbox computed from polygon: `(min(xs), min(ys), max(xs), max(ys))` — axis-aligned bounding box from the 4-point quadrilateral. Note: for rotated text, this bbox will be larger than the actual text region; acceptable for highlighting.
- Character offset range computed from the cumulative `"\n".join(rec_texts)`.

**Chandra (VLM):**
- Deferred to v2. Returns empty TextBlocks initially.
- Future approach: parse `layout_html` block elements into TextBlocks. The `layout_html` field is already captured on `PageResult` — the data is available when needed.

**Plain text (.txt, .md):**
- No TextBlocks (empty list). No source coordinates possible. Provenance records will have `start`/`end` offsets but no `page`/`bbox`.

### Storage

```python
@dataclass
class LoadedDocument:
    # ...existing fields unchanged...
    text_blocks: list[TextBlock] = field(default_factory=list)
    page_dimensions: list[tuple[float, float]] = field(default_factory=list)
```

`text_blocks` is built during document assembly. `page_dimensions` stores `(width, height)` in PDF points (72 dpi) per page, for the GUI to set up its canvas.

No changes to `PageResult` — blocks are assembled at the `LoadedDocument` level with offsets adjusted for page joins (`"\n\n"` separators).

### Lookup

```python
def locate_in_document(
    doc: LoadedDocument, start: int, end: int
) -> list[dict] | None:
    """Find page + bbox for a character range.

    Uses binary search over text_blocks. If the range spans multiple
    pages, returns one entry per page. Returns None if no blocks
    cover the range (e.g., plain text documents).

    Returns:
        [{"page": 1, "bbox": (x0, y0, x1, y1), "start": 342, "end": 400},
         {"page": 2, "bbox": (x0, y0, x1, y1), "start": 401, "end": 450}]
    """
```

Multi-page spans return a list of per-page segments — a single union bbox across pages is meaningless.

## Layer 2: Pipeline — Evidence Step

### The Problem with Multi-Step Pipelines

Radiology, pathology, and summarizer are multi-step chains (5+ DSPy signatures each). There is no single "final signature" to add an evidence field to. Each step operates on different text (e.g., `ExtractImpression` takes `impression_text`, not the full document).

### Strategy: Post-Extraction Evidence Gathering

A dedicated DSPy signature that runs **after** the pipeline completes:

```python
class GatherEvidence(dspy.Signature):
    """Given a source document and extracted fields, find the verbatim
    excerpt from the source document that supports each field value."""

    document_text: str = dspy.InputField(
        desc="Full text of the source document"
    )
    extracted_fields: str = dspy.InputField(
        desc="JSON of field names and their extracted values"
    )
    evidence: str = dspy.OutputField(
        desc="JSON mapping each field name to the verbatim excerpt "
             "from document_text that supports its value"
    )
```

This adds exactly **one LLM call** per pipeline invocation, regardless of how many steps the pipeline has. It sees the full source document and the complete extraction, so it can find the right excerpts.

**Applied to each pipeline:**
- **DocumentExtractor** — after `forward()` returns the extracted model, run `GatherEvidence` with `doc.text` + flattened field values.
- **Radiology/Pathology** — after `forward()` assembles findings/sections, run `GatherEvidence` with `doc.text` + all output fields.
- **Summarizer** — takes multiple source documents. Run `GatherEvidence` **per source document** — each summary point traces back to a specific document + position. The evidence map uses composite keys: `"doc_0:admission_date"`, `"doc_1:discharge_diagnosis"`, etc. Each provenance record includes a `document_index` field so the GUI knows which source file to highlight.
- **Deidentifier** — does NOT use `GatherEvidence`. The redaction map already has character offsets from the diff algorithm. Just needs coordinate lookup via `locate_in_document()`.

### Opt-In Flag

The evidence step only runs when provenance is requested:
- CLI: `--provenance` flag on `extract`, `deidentify`, and mode-specific commands
- SDK: `provenance=True` parameter
- MCP: `provenance` parameter on tools

When provenance is not requested, output is identical to today. Zero overhead.

### Excluded Pipelines

The following pipelines do not produce structured extractions from source documents. Provenance does not apply to them:
- `query_qa.py` — answers questions about documents, no field extraction.
- `verify_claim.py` — verifies claims, no field extraction.
- `schema_gen.py` — generates schemas from descriptions, no document input.
- `scaffold.py` — internal scaffolding utility.

## Layer 3: Provenance Resolution

New file: `mosaicx/pipelines/provenance.py`

### Core Function

```python
def resolve_provenance(
    doc: LoadedDocument,
    evidence: dict[str, str],
) -> dict[str, dict]:
    """Resolve evidence excerpts to source coordinates.

    For each field, finds the excerpt in the source text and looks up
    page/bbox coordinates from the document's TextBlock map.

    Returns a dict mapping field names to provenance records.
    """
```

### Resolution Strategy (3 Tiers)

1. **Exact match** — `doc.text.find(excerpt)`. Fast, deterministic. Used when the LLM quotes verbatim. If multiple exact matches exist, use the first unmatched occurrence (track which positions have been claimed).

2. **Fuzzy match** — `difflib.SequenceMatcher` best substring match. Threshold varies by excerpt length:
   - Excerpts >= 40 chars: 0.80 similarity
   - Excerpts < 40 chars: 0.90 similarity (higher threshold for short strings to avoid false positives in repetitive medical text like "No evidence of...")
   - If multiple matches exceed threshold, pick the closest unmatched occurrence.
   - If ambiguous (2+ matches with near-identical scores), mark as `"ambiguous"` rather than guessing wrong.

3. **Unresolved** — if no match above threshold, return `{"resolution": "unresolved"}`. The GUI knows not to render a highlight.

### Provenance Record Schema

```json
{
  "excerpt": "Left ventricular ejection fraction estimated at 55%",
  "start": 342,
  "end": 393,
  "spans": [
    {"page": 1, "bbox": [120.5, 445.2, 410.3, 458.7]}
  ],
  "resolution": "exact"
}
```

`resolution` is one of: `"exact"`, `"fuzzy"`, `"ambiguous"`, `"unresolved"`.

`spans` is a list because an excerpt may cross page boundaries. For single-page excerpts, this list has one entry.

When no TextBlocks are available (plain text input), `spans` is an empty list but `start`/`end` are still populated.

## Unified Output Format

Provenance data is added at the top level as `provenance` (user-facing data). Document metadata (`page_dimensions`) is added to the existing `_mosaicx._document` envelope.

**Naming note:** The `_mosaicx` envelope may contain a `provenance_requested` boolean flag (was provenance requested?). The top-level `provenance` dict contains the actual provenance records. These are distinct: the flag is metadata, the dict is output data.

### Extract / Radiology / Pathology / Summarizer

```json
{
  "extracted": {
    "ejection_fraction": "55%",
    "wall_motion": "normal"
  },
  "provenance": {
    "ejection_fraction": {
      "excerpt": "LV ejection fraction estimated at 55%",
      "start": 342, "end": 393,
      "spans": [{"page": 1, "bbox": [120.5, 445.2, 410.3, 458.7]}],
      "resolution": "exact"
    },
    "wall_motion": {
      "excerpt": "Wall motion is normal in all segments",
      "start": 510, "end": 547,
      "spans": [{"page": 1, "bbox": [90.0, 520.1, 400.0, 533.5]}],
      "resolution": "fuzzy"
    }
  },
  "_mosaicx": {
    "_document": {
      "path": "report.pdf",
      "pages": 2,
      "page_dimensions": [[612, 792], [612, 792]]
    }
  }
}
```

### Deidentify

```json
{
  "redacted_text": "Patient Name: [REDACTED] ...",
  "redaction_map": [
    {
      "original": "Jane Doe",
      "replacement": "[REDACTED]",
      "phi_type": "NAME",
      "method": "llm",
      "excerpt": "Patient Name: Jane Doe",
      "start": 14, "end": 22,
      "spans": [{"page": 1, "bbox": [276, 172, 357, 189]}],
      "resolution": "exact"
    }
  ],
  "_mosaicx": {
    "_document": {
      "path": "report.pdf",
      "pages": 1,
      "page_dimensions": [[612, 792]]
    }
  }
}
```

## SDK: Preserving LoadedDocument

Currently, `_resolve_documents()` in `sdk.py` extracts `doc.text` into a tuple and the `LoadedDocument` is garbage collected. To thread coordinate data through:

- Change `_resolve_documents()` to return `list[tuple[str, str, dict, LoadedDocument]]` — adding the full `LoadedDocument` as a fourth element.
- `_extract_single_text()`, `_deidentify_single_text()`, and related internal functions receive the `LoadedDocument` when `provenance=True`.
- When `provenance=False`, the `LoadedDocument` is not needed and can be discarded as today (no memory overhead).

## File Changes

| File | Change | Risk |
|------|--------|------|
| `documents/models.py` | Add `TextBlock`, add `text_blocks` + `page_dimensions` to `LoadedDocument` | Low — additive |
| `documents/loader.py` | Build TextBlocks in `_try_native_pdf_text()` and `_assemble_document()` | Medium — core path |
| `documents/engines/paddleocr_engine.py` | Extract `dt_polys`, store in PageResult metadata dict | Low — additive |
| `pipelines/provenance.py` | **New** — `GatherEvidence` signature, `resolve_provenance()`, `locate_in_document()`, fuzzy match | Low — new code |
| `pipelines/deidentifier.py` | Add coordinate lookup to redaction map entries when provenance requested | Low — extends existing |
| `pipelines/extraction.py` | Call `GatherEvidence` + `resolve_provenance()` after `forward()` returns, outside the dynamic signature machinery. Does not modify the dynamic signature construction. | Medium — integration point after forward() |
| `pipelines/radiology.py` | Call `GatherEvidence` + `resolve_provenance()` after `forward()` | Medium — evidence step added |
| `pipelines/pathology.py` | Call `GatherEvidence` + `resolve_provenance()` after `forward()` | Medium — evidence step added |
| `pipelines/summarizer.py` | Call `GatherEvidence` + `resolve_provenance()` after `forward()` | Medium — evidence step added |
| `report.py` | Thread `LoadedDocument` through to pipeline when provenance requested | Low — plumbing |
| `cli.py` | Add `--provenance` flag, include provenance in output, extend `_mosaicx` envelope | Low — output formatting |
| `sdk.py` | Add `provenance` param, preserve `LoadedDocument` in `_resolve_documents()`, extend `_build_document_meta()` with `page_dimensions`, include provenance in return | Medium — refactor `_resolve_documents()` + meta builder |
| `mcp_server.py` | Add `provenance` param to tools, include provenance in output | Low — output formatting |

## What Does NOT Change

- Existing Pydantic schemas in `schemas/` — untouched.
- Existing test datasets and evaluation — unchanged.
- Template system — unchanged.
- Optimization system — unchanged.
- `query_qa.py` and `verify_claim.py` — not applicable.
- Default behavior when `provenance` is not requested — identical to today.

## Testing Strategy

1. **TextBlock construction** — unit tests per backend: given a known PDF, verify TextBlocks have correct offsets and bboxes.
2. **locate_in_document()** — unit tests: given TextBlocks, verify lookups return correct page/bbox for various ranges, including multi-page spans.
3. **Critical invariant** — test that `doc.text` passed to pipeline is identical to the text used to build TextBlocks. No intermediate modification.
4. **resolve_provenance()** — unit tests: exact match, fuzzy match, ambiguous, unresolved cases. Include repetitive medical text to verify disambiguation.
5. **GatherEvidence** — integration test: run on sample PDF extraction, verify evidence excerpts are verbatim from source.
6. **Deidentifier enrichment** — test coordinate lookup on sample PDF redaction map.
7. **End-to-end** — run extract + deidentify on sample PDF with `provenance=True`, verify output has valid coordinates that map back to correct regions.
8. **Regression** — all existing tests must pass. Provenance is additive and opt-in.

## Implementation Order

1. `TextBlock` model + `locate_in_document()` + tests
2. pypdfium2 TextBlock builder + tests
3. PaddleOCR TextBlock builder + tests
4. `GatherEvidence` signature + `resolve_provenance()` + fuzzy matching + tests
5. SDK `_resolve_documents()` refactor to preserve `LoadedDocument`
6. Deidentifier coordinate enrichment + tests
7. Extraction pipeline evidence integration + tests
8. Radiology/pathology/summarizer evidence integration + tests
9. CLI `--provenance` flag + output threading
10. MCP provenance parameter + output threading
11. Full integration test on sample PDF

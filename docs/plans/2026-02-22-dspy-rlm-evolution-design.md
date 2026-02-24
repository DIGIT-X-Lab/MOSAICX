# MOSAICX Evolution: DSPy 3.x + RLM Integration Design

**Date:** 2026-02-22
**Status:** Historical design reference (non-canonical)
**Scope:** Architecture redesign to add Verify, Query, and DSPy 3.x upgrades
**Canonical Status Board:** `docs/plans/2026-02-24-roadmap-status-audit.md`

---

## Vision

MOSAICX becomes the open-source, local-first medical document intelligence library.

```
Extract. Verify. Query. Trust.
```

Four capabilities, two cross-cutting concerns:

| Capability | Status | Engine |
|---|---|---|
| **Extract** | Existing, upgraded | DSPy Signatures + Refine/BestOfN |
| **Verify** | New | Deterministic checks + LLM + RLM |
| **Query** | New | RLM (conversational, multi-source) |
| **Template** | Existing, expanded | SchemaGenerator + YAML compiler |

| Cross-cutting | Description |
|---|---|
| **Metadata envelope** (`_mosaicx`) | Always-on: version, model, pipeline, timing, document info |
| **Provenance** (`_provenance`) | Opt-in: per-field source evidence with text excerpts and locations |

## Problem Statement

Healthcare organizations are adopting LLMs for document processing, but face
three unsolved problems:

1. **No verification.** LLM outputs are trusted blindly. A wrong measurement
   (12mm vs 22mm) can change clinical decisions. No tool systematically
   verifies extracted data against source documents.

2. **No cross-document intelligence.** After extracting 500 reports, there is
   no way to ask "how many patients had EF < 40%?" without writing custom
   scripts. Researchers need conversational access to structured and
   unstructured medical data.

3. **Data sovereignty.** Patient data cannot leave hospital networks. Cloud-only
   solutions (AWS HealthLake, Google Healthcare API) are not viable for many
   institutions. Local-first, open-source tooling is required.

MOSAICX solves all three by combining DSPy's optimizable pipelines with RLM's
programmatic document exploration, purpose-built for healthcare.

## Target Users

| User | Primary capability | Example workflow |
|---|---|---|
| Hospital data scientist | Extract + Verify | Batch-extract 10,000 radiology reports, audit accuracy |
| RAG application builder | Verify (standalone) | Validate chatbot answers against source documents |
| Clinical researcher | Query | "Show me tumor size trends for patients on immunotherapy" |
| Compliance officer | Verify + Query | Audit trail: what evidence supports each data point? |

---

## Architecture Overview

```
mosaicx/
├── pipelines/              # Existing extraction pipelines (DSPy Modules)
│   ├── radiology.py        # Upgraded: Refine/BestOfN wrappers
│   ├── pathology.py        # Upgraded: Refine/BestOfN wrappers
│   ├── extraction.py       # Upgraded: Refine/BestOfN wrappers
│   ├── summarizer.py       # Upgraded: RLM option for large inputs
│   ├── deidentifier.py     # Unchanged
│   └── schema_gen.py       # Unchanged
│
├── verify/                 # NEW: Verification engine
│   ├── engine.py           # Orchestrator: routes to correct verification level
│   ├── deterministic.py    # Level 1: regex, value matching, schema checks
│   ├── spot_check.py       # Level 2: single LLM call for high-risk fields
│   └── audit.py            # Level 3: full RLM-based cross-referencing
│
├── query/                  # NEW: Conversational query engine
│   ├── engine.py           # RLM wrapper with MOSAICX tools
│   ├── session.py          # Stateful session management
│   ├── tools.py            # MOSAICX-specific RLM tools
│   └── loaders.py          # Source loaders (PDF, JSON, parquet, Excel, CSV)
│
├── provenance/             # NEW: Source tracking
│   ├── models.py           # FieldEvidence, SourceSpan, ProvenanceMap
│   ├── inline.py           # Extended DSPy Signatures with evidence fields
│   └── resolve.py          # Resolve char offsets from LLM excerpts
│
├── envelope.py             # NEW: _mosaicx metadata builder
│
├── schemas/                # Existing template system
│   └── radreport/templates/  # Expanded: pathology templates, more radiology
│
├── evaluation/             # Upgraded optimization
│   ├── optimize.py         # GEPA/SIMBA support
│   └── metrics.py          # Medical-domain feedback for GEPA
│
├── sdk.py                  # Extended: verify(), query(), provenance flag
├── cli.py                  # Extended: verify, query, --provenance flag
└── mcp_server.py           # Extended: verify, query, provenance tools
```

---

## Capability 1: Extract (Upgraded)

### Current state

Six DSPy pipelines using `dspy.Predict` and `dspy.ChainOfThought`. Works well
for short-to-medium documents. No self-healing on extraction errors.

### Upgrades

#### 1.1 Self-healing with dspy.Refine

Wrap critical extraction steps in `dspy.Refine` to automatically retry with
feedback when output quality is low.

```python
# Before:
self.extract_findings = dspy.ChainOfThought(ExtractRadFindings)

# After:
def _findings_reward(args, pred):
    score = 0.0
    findings = pred.findings
    if findings:
        score += 0.3
    for f in findings:
        if f.anatomy and f.anatomy.strip():
            score += 0.1
    if any(f.measurement for f in findings):
        score += 0.2
    return min(score, 1.0)

self.extract_findings = dspy.Refine(
    module=dspy.ChainOfThought(ExtractRadFindings),
    N=3,
    reward_fn=_findings_reward,
    threshold=0.7,
)
```

**Where to apply Refine:**
- `ExtractRadFindings` (radiology) -- most complex extraction step
- `ExtractImpression` (radiology) -- often returns empty
- `ExtractPathDiagnosis` (pathology) -- highest clinical importance
- `DocumentExtractor` in schema mode -- template compliance

**Where NOT to apply (overkill):**
- `ClassifyExamType` -- simple classification, Predict is fine
- `ParseReportSections` -- structural parsing, deterministic-ish
- `ExtractTechnique` -- low stakes, simple fields

#### 1.2 BestOfN for critical fields

For the highest-stakes fields (measurements, diagnosis), use `dspy.BestOfN` to
run multiple extractions and pick the best.

```python
self.extract_diagnosis = dspy.BestOfN(
    module=dspy.ChainOfThought(ExtractPathDiagnosis),
    N=5,
    reward_fn=diagnosis_reward,
    threshold=0.8,
)
```

#### 1.3 JSONAdapter for structured output

Switch from default `ChatAdapter` to `JSONAdapter` for extraction steps that
output Pydantic models. This leverages native JSON mode in compatible LLMs
(Ollama supports this) for more reliable structured output.

```python
import dspy
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
```

**Note:** JSONAdapter should be opt-in per-pipeline, not global. Some steps
(narrative synthesis) work better with ChatAdapter.

#### 1.4 dspy.Reasoning for thinking models

When using reasoning-capable models via Ollama (DeepSeek R1, QwQ), capture
native thinking tokens for complex extraction steps.

```python
class ExtractRadFindings(dspy.Signature):
    report_text: str = dspy.InputField()
    reasoning: dspy.Reasoning = dspy.OutputField()  # captures thinking
    findings: list[RadReportFinding] = dspy.OutputField()
```

This is model-dependent -- only activate when the configured LM supports
reasoning tokens. Add a config flag: `MOSAICX_USE_REASONING=true`.

#### 1.5 Optimizer upgrades

**GEPA with medical feedback:**

GEPA's textual feedback mechanism is ideal for medical extraction because
diagnostic feedback can guide prompt evolution:

```python
def medical_metric(gold, pred, trace, pred_name, pred_trace):
    score = compute_f1(gold.findings, pred.findings)
    feedback = []
    if missing := find_missing_fields(gold, pred):
        feedback.append(f"Missed: {missing}")
    if wrong := find_incorrect_values(gold, pred):
        feedback.append(f"Wrong values: {wrong}")
    return {"score": score, "feedback": "; ".join(feedback)}
```

**SIMBA for small datasets:**

Medical labeled data is scarce. SIMBA's sample-efficient optimization is
better suited than MIPROv2 when you only have 20-50 labeled examples.

**Updated budget presets:**

| Budget | Optimizer | Estimate |
|---|---|---|
| light | BootstrapFewShot | ~5 min |
| medium | SIMBA | ~15 min |
| heavy | GEPA | ~45 min |

---

## Capability 2: Verify (New)

### Purpose

Verify any healthcare LLM output against source documents. Works for MOSAICX's
own extractions AND for external systems (RAG chatbots, other extraction tools).

### Verification types

| Error type | Description | Example |
|---|---|---|
| Value mismatch | Extracted value differs from source | EF: 45% extracted, source says 55% |
| Hallucination | Claim not found in any source | "History of diabetes" -- no mention in docs |
| Omission | Important context dropped | "Stable" -- but source adds "recommend 3-month follow-up" |
| Attribution error | Right fact, wrong source/date | "January CT showed improvement" -- was actually June CT |
| Temporal error | Wrong sequence of events | "After chemo, tumor shrank" -- scan predates chemo |

### Three verification levels

#### Level 1: Deterministic (< 1 second, no LLM)

```python
# mosaicx/verify/deterministic.py

def verify_deterministic(extraction: dict, source_text: str) -> VerificationReport:
    issues = []

    # Check measurements exist in source
    for finding in extraction.get("findings", []):
        if m := finding.get("measurement"):
            value_str = f"{m['value']}"
            if value_str not in source_text:
                issues.append(Issue(
                    type="value_not_found",
                    field=f"findings[{i}].measurement",
                    claimed=value_str,
                    severity="warning",
                ))

    # Check enum values are valid
    # Check finding_refs point to valid indices
    # Check internal consistency (modality vs technique)
    ...
```

**Use case:** Real-time middleware in RAG pipelines. Check every response
before showing to user. Catches obvious errors instantly.

#### Level 2: LLM spot-check (3-10 seconds, single LLM call)

```python
# mosaicx/verify/spot_check.py

class VerifyClaim(dspy.Signature):
    """Verify whether claims are supported by the source text."""
    source_text: str = dspy.InputField(desc="Original document text")
    claims: str = dspy.InputField(desc="JSON list of claims to verify")
    verdicts: list[ClaimVerdict] = dspy.OutputField(
        desc="Verdict for each claim: supported/contradicted/unsupported"
    )
```

Selects the 3-5 highest-risk fields (measurements, diagnosis, severity) and
asks a single focused LLM call to verify them against the source.

**Use case:** Post-extraction quality check. Run after every extraction,
flag suspicious results for human review.

#### Level 3: RLM audit (30-90 seconds, multi-step)

Full programmatic cross-referencing via RLM. The model writes Python code to
systematically verify every claim:

1. Parse extraction into individual claims
2. For each claim, search source document(s) for supporting evidence
3. Cross-reference values (measurements, dates, terminology)
4. Check for omissions (important source content not in extraction)
5. Verify temporal relationships across multiple documents
6. Produce a detailed audit report with evidence chains

**Use case:** Research dataset validation, clinical decision support,
regulatory compliance.

### Verification output

```python
@dataclass
class VerificationReport:
    verdict: Literal["verified", "partially_supported", "contradicted", "unverifiable"]
    confidence: float  # 0.0 - 1.0
    level: Literal["deterministic", "spot_check", "audit"]
    field_verdicts: dict[str, FieldVerdict]
    issues: list[Issue]
    evidence: list[Evidence]
    missed_content: list[str]  # important source content not in extraction

@dataclass
class FieldVerdict:
    status: Literal["correct", "mismatch", "unsupported", "not_checked"]
    claimed_value: str | None
    source_value: str | None
    evidence_excerpt: str | None
    severity: Literal["info", "warning", "critical"]

@dataclass
class Issue:
    type: str  # value_mismatch, hallucination, omission, attribution, temporal
    field: str
    detail: str
    severity: Literal["info", "warning", "critical"]

@dataclass
class Evidence:
    source: str  # filename or "source_text"
    excerpt: str
    supports: str | None
    contradicts: str | None
```

### API surface

**SDK:**

```python
import mosaicx

# Verify MOSAICX's own extraction
extraction = mosaicx.extract(documents="report.pdf", template="chest_ct")
report = mosaicx.verify(
    extraction=extraction,
    sources=["report.pdf"],
    level="standard",          # "quick" | "standard" | "audit"
)

# Verify external RAG output
report = mosaicx.verify(
    claim="The ejection fraction was 45% on the last echo.",
    sources=["echo_2025.pdf", "echo_2024.pdf"],
    level="audit",
)

# Batch verify
reports = mosaicx.verify_batch(
    claims=claims_list,
    sources=source_docs,
)

# Integrated extract + verify
result = mosaicx.extract(
    documents="report.pdf",
    template="chest_ct",
    verify=True,               # auto-verify after extraction
)
# result["_verification"] = VerificationReport
```

**CLI:**

```bash
# Verify a claim against sources
mosaicx verify --claim "EF was 45%" --sources echo.pdf

# Verify a MOSAICX extraction (pipe-friendly)
mosaicx extract --document report.pdf --template chest_ct --verify

# Full audit
mosaicx verify --claim "..." --sources docs/ --level audit

# Batch audit extracted results against originals
mosaicx verify --extractions ./output/ --sources ./originals/
```

**MCP:**

```python
@mcp.tool()
def verify_output(
    claim: str | None = None,
    extraction: dict | None = None,
    source_text: str = "",
    level: str = "standard",
) -> str:
    """Verify a healthcare LLM output against source documents."""
```

---

## Capability 3: Query (New)

### Purpose

Conversational Q&A over any combination of medical data: raw documents,
extracted JSON, parquet files, Excel sheets, CSV files. Powered by RLM
with MOSAICX tools available in the sandbox.

### Design principles

1. **Load anything** -- PDFs, JSON, parquet, Excel, CSV, SQL (future)
2. **Conversational** -- stateful sessions with follow-up questions
3. **Full data science** -- text answers + structured data + artifacts (charts, exports)
4. **MOSAICX-integrated** -- can extract and verify on-the-fly inside queries
5. **Two-model architecture** -- smart model writes code, cheap model handles sub-queries

### Session lifecycle

```
mosaicx.query(sources=[...])
    │
    ├── LOAD: documents, JSON, DataFrames
    ├── INDEX: build catalog with metadata
    ├── CONFIGURE: RLM with MOSAICX tools
    │
    ├── session.ask("question 1")
    │   ├── RLM receives: catalog + question + history
    │   ├── RLM writes Python code
    │   ├── Code executes in sandbox
    │   └── Returns: {answer, data, artifacts}
    │
    ├── session.ask("question 2")  # builds on prior context
    │   └── ...
    │
    └── session.close()
```

### Source loading

```python
# mosaicx/query/loaders.py

SUPPORTED_SOURCES = {
    ".pdf": load_pdf,         # via mosaicx.documents.loader (OCR)
    ".txt": load_text,        # plain text
    ".md": load_text,         # markdown
    ".json": load_json,       # parsed to dict/list
    ".jsonl": load_jsonl,     # line-delimited JSON
    ".parquet": load_parquet, # pandas DataFrame
    ".xlsx": load_excel,      # pandas DataFrame
    ".xls": load_excel,       # pandas DataFrame
    ".csv": load_csv,         # pandas DataFrame
}
```

Each source becomes an entry in the session catalog:

```python
@dataclass
class SourceMeta:
    name: str                  # filename or user-provided name
    source_type: str           # "document", "json", "dataframe"
    format: str                # "pdf", "parquet", etc.
    size: int                  # rows for tables, chars for text
    schema: dict | None        # column names + types for DataFrames
    preview: str               # first 500 chars or first 5 rows
```

### RLM configuration

```python
import dspy

rlm = dspy.RLM(
    signature="catalog, question, history -> answer",
    max_iterations=20,
    max_llm_calls=50,
    max_output_chars=10000,
    tools=[
        mosaicx_extract,
        mosaicx_verify,
        search_documents,
        get_document,
        get_dataframe,
        save_artifact,
    ],
    sub_lm=dspy.LM(config.lm_cheap),  # cheap model for llm_query()
)
```

### Custom MOSAICX tools for RLM

```python
# mosaicx/query/tools.py

def mosaicx_extract(text: str, template: str | None = None) -> dict:
    """Extract structured data from text using a MOSAICX pipeline."""

def mosaicx_verify(claim: str, source_text: str) -> dict:
    """Verify a claim against source text."""

def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search loaded documents by keyword/semantic similarity."""

def get_document(name_or_index: str | int) -> str:
    """Get full text of a loaded document."""

def get_dataframe(name: str) -> str:
    """Get a loaded DataFrame (returns repr for RLM context)."""

def save_artifact(data: Any, filename: str, format: str = "csv") -> str:
    """Save data as a file artifact (CSV, JSON, PNG)."""
```

### Output types

```python
@dataclass
class QueryResponse:
    answer: str                           # natural language answer
    data: dict[str, Any] | None           # structured data (counts, stats)
    artifacts: list[Artifact]             # files (CSV, charts, exports)
    verification: VerificationReport | None  # if self-verified
    code_executed: str | None             # the Python code RLM ran (for transparency)

@dataclass
class Artifact:
    type: str       # "csv", "json", "image", "parquet", "pdf"
    path: str       # file path
    description: str
```

### Fast path (skip RLM)

For simple questions answerable from catalog metadata:

- "How many files did I load?" -> catalog lookup
- "What columns are in the parquet?" -> schema lookup
- "Show me the first 5 rows" -> DataFrame.head()

These don't need the full RLM REPL, saving 10-30 seconds.

### API surface

**SDK:**

```python
import mosaicx

session = mosaicx.query(
    sources=["reports/*.pdf", "registry.parquet", "extractions/"],
    template="chest_ct",                    # optional
    sub_lm="openai/gpt-oss:20b",           # optional
)

r1 = session.ask("How many patients are in this dataset?")
# → QueryResponse(answer="47 patients across 142 reports", data={...})

r2 = session.ask("What percentage had pulmonary nodules > 6mm?")
# → QueryResponse(answer="23.4%", data={"count": 11, "pct": 23.4})

r3 = session.ask("Plot the size distribution")
# → QueryResponse(artifacts=[Artifact(type="image", path="...")])

r4 = session.ask("Export nodule patients as CSV")
# → QueryResponse(artifacts=[Artifact(type="csv", path="...")])

session.close()
```

**CLI:**

```bash
$ mosaicx query --sources reports/ registry.parquet

  Loaded: 142 PDF reports, 1 parquet (2847 rows)

> how many patients had EF < 40%?
  12 out of 47 patients (25.5%)

> export them as CSV
  Saved to ./low_ef_patients.csv (12 rows)
```

**MCP:**

```python
@mcp.tool()
def query_start(sources: list[str], template: str | None = None) -> str:
    """Start a conversational query session over medical data."""

@mcp.tool()
def query_ask(session_id: str, question: str) -> str:
    """Ask a question in an active query session."""

@mcp.tool()
def query_close(session_id: str) -> str:
    """Close a query session and free resources."""
```

---

## Capability 4: Template (Existing, Expanded)

### No changes to core template system

The YAML template format, compiler, resolution logic, and SchemaGenerator
pipeline remain unchanged. Templates become more valuable with Verify and
Query because they define what "correct" looks like.

### Planned expansions

- **More built-in templates:** Add pathology templates (currently only
  radiology). Surgical pathology, cytology, molecular pathology.
- **Template-guided verification:** When verifying, use the template to know
  which fields are high-risk (required fields, measurements, diagnoses).
- **Template-guided query:** When querying raw documents, use templates to
  understand document structure without requiring explicit extraction.

---

## DSPy 3.x Features Summary

| Feature | Where used | Priority |
|---|---|---|
| `dspy.Refine` | Extract pipelines (findings, diagnosis steps) | High |
| `dspy.BestOfN` | Extract pipelines (critical fields) | High |
| `dspy.RLM` | Verify (level 3), Query engine | High |
| `JSONAdapter` | Extract pipelines (structured output steps) | Medium |
| GEPA optimizer | Evaluation/optimization system | Medium |
| SIMBA optimizer | Evaluation/optimization (small datasets) | Medium |
| `dspy.Reasoning` | Extract (when using thinking models) | Low |
| `dspy.Parallel` | Batch extraction parallelization | Low |
| ArborGRPO | Not applicable (requires GPU finetuning) | N/A |

---

## Dependencies

### New runtime dependencies

| Dependency | Required for | Notes |
|---|---|---|
| deno | RLM sandbox (Pyodide WASM) | Required for Query and Verify level 3 |
| pandas | Query (DataFrame operations) | Already common in data science |
| openpyxl | Query (Excel loading) | Optional, for .xlsx support |
| matplotlib | Query (chart artifacts) | Optional, for plotting |

### Dependency strategy

RLM/Query features require Deno for the sandbox. Install as optional extra:

```toml
[project.optional-dependencies]
query = ["pandas", "openpyxl", "matplotlib"]
```

Deno is a system dependency, documented in installation instructions.
Extract and Verify levels 1-2 work without Deno.

---

## Phased Rollout

### Phase 1: Metadata envelope + self-healing extraction

- Implement `_mosaicx` metadata envelope on all outputs (replaces `_metrics`/`_document`)
- Add `dspy.Refine` wrappers to radiology, pathology, extraction pipelines
- Add `dspy.BestOfN` for high-stakes fields
- Add GEPA/SIMBA to optimizer budget presets
- Update evaluation metrics with textual feedback for GEPA
- No new dependencies, no architectural changes

### Phase 2: Provenance + Verification engine

- Implement inline provenance (`--provenance` flag)
- Extended DSPy Signatures with evidence output fields

- Implement `mosaicx/verify/` module (all 3 levels)
- Add `mosaicx.verify()` to SDK
- Add `mosaicx verify` CLI command
- Add `verify_output` MCP tool
- Add `--verify` flag to `mosaicx extract`
- Levels 1-2 work without Deno; level 3 requires Deno

### Phase 3: Query engine

- Implement `mosaicx/query/` module
- Add `mosaicx.query()` to SDK
- Add `mosaicx query` CLI command (interactive)
- Add `query_start/ask/close` MCP tools
- Requires Deno for RLM sandbox
- Source loaders for JSON, parquet, Excel, CSV

### Phase 4: Polish and expand

- More built-in templates (pathology, cardiology)
- `dspy.Reasoning` support for thinking models
- SQLite/DuckDB source support for Query
- Query session persistence (save/resume sessions)
- Performance optimization (fast path routing, caching)

---

## Cross-cutting: Metadata Envelope + Provenance

### Metadata envelope (`_mosaicx`)

Every MOSAICX output — extract, verify, query, deidentify, summarize —
includes a standard `_mosaicx` metadata block. Always present, not optional.

```json
{
  "exam_type": "CT Chest",
  "findings": ["..."],

  "_mosaicx": {
    "version": "2.1.0",
    "pipeline": "radiology",
    "template": "chest_ct",
    "template_version": "1.0.0",
    "model": "openai/gpt-oss:120b",
    "model_temperature": 0.0,
    "timestamp": "2026-02-22T14:32:01Z",
    "duration_s": 12.4,
    "tokens": {"input": 2340, "output": 890},
    "provenance": false,
    "verification": null,
    "document": {
      "file": "report.pdf",
      "pages": 2,
      "ocr_engine": "surya",
      "quality_warning": null
    }
  }
}
```

**Purpose:** Reproducibility and auditability. A researcher can report: "Data
extracted using MOSAICX 2.1.0, model gpt-oss:120b, template chest_ct v1.0.0."
A compliance officer can verify: "This extraction was verified at confidence
0.92 on 2026-02-22."

**Design rules:**
- `_mosaicx` is always present on every output dict
- Fields that don't apply are `null`, never omitted
- Replaces the current ad-hoc `_metrics` and `_document` keys
- Serializable as JSON (no Python objects)

### Provenance tracking (`_provenance`)

Opt-in (`--provenance` flag or `provenance=True`). When enabled, the LLM
provides source evidence for each extracted field during extraction (inline
approach). Every extracted value links back to the exact text in the source
document that supports it.

```json
{
  "findings": [
    {
      "anatomy": "C6-7",
      "severity": "mild",
      "description": "At C6-7 there is mild bilateral bony neural foraminal narrowing..."
    }
  ],

  "_provenance": {
    "findings[0].severity": {
      "source_excerpt": "there is mild bilateral bony neural foraminal narrowing",
      "source_location": {"page": 1, "line_start": 14, "char_start": 847, "char_end": 912},
      "confidence": 0.95
    },
    "findings[0].anatomy": {
      "source_excerpt": "At C6-7",
      "source_location": {"page": 1, "line_start": 14, "char_start": 841, "char_end": 847},
      "confidence": 0.98
    },
    "findings[0].description": {
      "source_excerpt": "At C6-7 there is mild bilateral bony neural foraminal narrowing without central canal compromise.",
      "source_location": {"page": 1, "line_start": 14, "char_start": 841, "char_end": 938},
      "confidence": 0.99
    }
  }
}
```

**Implementation: inline extraction.**

DSPy Signatures are extended to request evidence alongside extracted values:

```python
class FieldEvidence(BaseModel):
    field_path: str          # e.g. "findings[0].severity"
    source_excerpt: str      # exact text from source
    source_location: SourceSpan | None  # char offsets if determinable
    confidence: float        # 0.0 - 1.0

class ExtractRadFindings(dspy.Signature):
    """Extract findings with source evidence."""
    report_text: str = dspy.InputField()
    findings: list[RadReportFinding] = dspy.OutputField()
    evidence: list[FieldEvidence] = dspy.OutputField(
        desc="For each finding field, the exact source text that supports it"
    )
```

**Why inline (not post-hoc):**
- The LLM knows WHERE it got the value during extraction -- asking after
  the fact requires re-reading the document and guessing.
- Inline evidence is more accurate for semantic extractions (e.g., "mild"
  extracted from a sentence that uses the word "modest").
- Post-hoc alignment fails when values are inferred rather than directly
  quoted (e.g., severity inferred from context).

**Provenance + Verification synergy:**

When provenance is enabled, verification becomes significantly cheaper:
- Level 1 (deterministic): check that `source_excerpt` actually appears at
  `source_location` in the document. Near-instant.
- Level 2 (spot-check): only verify fields where `confidence < 0.8` instead
  of spot-checking everything.
- Level 3 (RLM audit): use provenance as starting points for deeper
  cross-referencing rather than searching from scratch.

**API:**

```python
# SDK
result = mosaicx.extract(documents="report.pdf", template="chest_ct", provenance=True)

# CLI
mosaicx extract --document report.pdf --template chest_ct --provenance

# MCP
extract_document(document_text="...", template="chest_ct", provenance=true)
```

---

## What MOSAICX is NOT

To stay focused, MOSAICX should explicitly avoid:

- **Building a full RAG stack.** MOSAICX is not a vector database or retrieval
  framework. It processes, verifies, and queries documents -- not retrieval.
- **Clinical decision support.** MOSAICX extracts and verifies data. It does
  not make clinical recommendations or diagnoses.
- **EHR integration.** MOSAICX works with documents and data files, not
  HL7/FHIR feeds. EHR integration is a separate concern.
- **Training/finetuning models.** MOSAICX optimizes prompts (via DSPy), not
  model weights. ArborGRPO is explicitly out of scope.

---

## Success Criteria

MOSAICX is successful when:

1. A researcher can extract 1,000 reports and trust the output because
   verification caught errors before they entered the dataset.
2. A RAG builder can add `mosaicx.verify(claim, sources)` to their pipeline
   in 3 lines of code and catch hallucinations.
3. A clinician can load a patient folder and ask "what happened to the
   lung nodule?" and get a verified, evidence-backed answer.
4. All of this runs on a hospital workstation, local-first, with no data
   leaving the network.

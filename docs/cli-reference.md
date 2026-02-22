# CLI Reference

MOSAICX is controlled entirely from the command line. Every command supports `--help` for quick reference.

This reference covers all commands, flags, and options. Each command includes practical examples assuming you're starting fresh with no prior MOSAICX experience.

## Global Options

Available on all commands:

| Flag | Description |
|------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help message and exit |

**Examples:**
```bash
# Check your version
mosaicx --version

# Show main help
mosaicx --help

# Get help on a specific command
mosaicx extract --help
```

---

## `mosaicx extract`

Extract structured data from a clinical document.

MOSAICX can extract data in three ways:
1. **Auto mode** (no flags): LLM automatically determines what to extract
2. **Template mode** (`--template`): Use a built-in template, user template, YAML file, or legacy saved schema
3. **Mode mode** (`--mode`): Use a built-in multi-step pipeline (radiology, pathology)

The `--template` flag resolves its argument through a resolution chain:
1. YAML file path (if suffix is `.yaml`/`.yml` and file exists)
2. User template in `~/.mosaicx/templates/`
3. Built-in template name (e.g. `chest_ct`, `brain_mri`)
4. Legacy saved schema from `~/.mosaicx/schemas/`
5. Error if nothing matches

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--document` | PATH | Yes | Path to the document (PDF, TXT, DOCX, PNG, JPG, TIFF) |
| `--template` | TEXT | No | Template name, YAML file path, or saved schema name |
| `--mode` | TEXT | No | Extraction mode (e.g., `radiology`, `pathology`) |
| `--score` | flag | No | Score completeness of extracted data against the template |
| `--optimized` | PATH | No | Path to an optimized DSPy program (`.json` file) |
| `-o`, `--output` | PATH | No | Save output to JSON or YAML file |
| `--list-modes` | flag | No | List available extraction modes and exit |
| `--dir` | PATH | No | Directory of documents for batch processing |
| `--workers` | INT | No | Number of parallel workers (default: 1) |
| `--output-dir` | PATH | No | Directory for output files (batch mode) |
| `--format` | TEXT | No | Export format(s): `jsonl`, `parquet` (can repeat) |
| `--resume` | flag | No | Resume from last checkpoint |

**Important:**
- `--template` and `--mode` are mutually exclusive -- use only one
- `--document` and `--dir` are mutually exclusive -- use only one
- If neither `--template` nor `--mode` is provided, auto mode is used
- Supported formats: PDF, TXT, DOCX, MD, PNG, JPG, JPEG, TIF, TIFF

**Examples:**

```bash
# Auto mode -- LLM decides what to extract from the document
mosaicx extract --document report.pdf

# List available modes
mosaicx extract --list-modes

# Radiology mode -- 5-step pipeline for radiology reports
# Steps: classify exam -> parse sections -> extract technique -> findings -> impression
mosaicx extract --document ct_chest.pdf --mode radiology

# Pathology mode -- 5-step pipeline for pathology reports
# Steps: classify specimen -> parse sections -> specimen details -> findings -> diagnosis
mosaicx extract --document biopsy.pdf --mode pathology

# Use a built-in template by name
mosaicx extract --document ct_chest.pdf --template chest_ct

# Use a user-created YAML template file
mosaicx extract --document report.pdf --template echo.yaml

# Use a legacy saved schema (resolved from ~/.mosaicx/schemas/)
mosaicx extract --document echo.pdf --template EchoReport

# Extract with completeness scoring
mosaicx extract --document ct_chest.pdf --template chest_ct --score

# Save output to JSON
mosaicx extract --document report.pdf --mode radiology -o output.json

# Save output to YAML
mosaicx extract --document report.pdf --mode radiology -o output.yaml

# Use an optimized program (from mosaicx optimize)
mosaicx extract --document report.pdf --template chest_ct \
  --optimized ~/.mosaicx/optimized/radiology_optimized.json

# Combine mode with custom save location
mosaicx extract --document ct_report.pdf --mode radiology \
  -o /path/to/results/structured_report.json

# Batch process a directory
mosaicx extract --dir ./reports --output-dir ./structured --mode radiology

# Batch with 4 parallel workers
mosaicx extract --dir ./reports --output-dir ./structured --workers 4

# Batch with export formats
mosaicx extract --dir ./reports --output-dir ./structured --format jsonl --format parquet

# Resume a failed batch
mosaicx extract --dir ./reports --output-dir ./structured --resume
```

**What you'll see:**

Without `--output`, results are displayed in the terminal as formatted tables. Use `--output` to save the full structured data as JSON or YAML.

When `--score` is used, a completeness report is shown after the extracted data, scoring how thoroughly the template fields were populated.

---

## `mosaicx template create`

Create a new YAML template from a description, sample document, web page, RadReport ID, or JSON schema.

Templates are saved to `~/.mosaicx/templates/` by default and can be reused with `mosaicx extract --template`.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--describe` | TEXT | No* | Natural-language description of the template |
| `--from-document` | PATH | No* | Infer template from a sample document |
| `--from-url` | TEXT | No* | Infer template from a web page (e.g. RadReport URL) |
| `--from-radreport` | TEXT | No* | RadReport template ID (e.g. `RPT50890` or `50890`) |
| `--from-json` | PATH | No* | Convert a saved SchemaSpec JSON to YAML template |
| `--name` | TEXT | No | Override the template name (default: LLM-chosen) |
| `--mode` | TEXT | No | Pipeline mode to embed (e.g. `radiology`, `pathology`) |
| `--output` | PATH | No | Custom save path (default: `~/.mosaicx/templates/`) |

**Important:**
- Must provide at least one source: `--describe`, `--from-document`, `--from-url`, `--from-radreport`, or `--from-json`
- `--from-json` cannot be combined with other sources
- `--describe` and `--from-document` can be combined for better results
- Templates are saved as YAML files in `~/.mosaicx/templates/{name}.yaml`

**Examples:**

```bash
# Generate from description
mosaicx template create \
  --describe "echo report with LVEF, valve grades, chamber dimensions, and impression"

# Generate from sample document
mosaicx template create --from-document sample_echo.pdf

# Combine description and document
mosaicx template create \
  --describe "extract vital signs and lab values" \
  --from-document clinic_note.pdf

# Generate from a web page
mosaicx template create --from-url https://radreport.org/template/0050890

# Generate from a RadReport template ID
mosaicx template create --from-radreport RPT50890

# Convert a legacy JSON schema to YAML template
mosaicx template create --from-json ~/.mosaicx/schemas/EchoReport.json

# Override the auto-generated name
mosaicx template create \
  --describe "CT lung nodule report with LUNG-RADS score" \
  --name CTLungNodule

# Embed a pipeline mode in the template
mosaicx template create \
  --describe "chest CT report" --mode radiology

# Save to custom location
mosaicx template create \
  --describe "chest x-ray findings" \
  --output /path/to/my_templates/chest_xr.yaml
```

**What happens:**

1. LLM analyzes your description, document, or web content
2. Generates a YAML template with sections, types, and descriptions
3. Saves the template to `~/.mosaicx/templates/{name}.yaml`
4. Displays a preview of the generated YAML

You can now use the template with:
```bash
mosaicx extract --document new_echo.pdf --template EchoReport
```

---

## `mosaicx template list`

List available built-in and user-created templates.

Built-in templates are pre-defined YAML schemas for common radiology exams. User templates are stored in `~/.mosaicx/templates/`.

**Examples:**

```bash
mosaicx template list
```

**Output:**

Shows two tables:

1. **Built-in Templates** -- with columns:
   - Template name
   - Mode (e.g., radiology)
   - RDES (RadReport ID, if applicable)
   - Description

2. **User Templates** (if any exist) -- with columns:
   - Template name
   - Description

---

## `mosaicx template show`

Display details of a template (built-in, user-created, or legacy saved schema).

**Usage:**

```bash
mosaicx template show <name>
```

**Examples:**

```bash
# Show a built-in template
mosaicx template show chest_ct

# Show a user-created template
mosaicx template show EchoReport

# Show a legacy saved schema
mosaicx template show CTLungNodule
```

**Output:**

Displays:
- Template name and source (built-in or user)
- Description
- Mode and RDES ID (if applicable)
- Table of sections/fields with name, type, required status, and description

---

## `mosaicx template refine`

Refine an existing template using LLM-powered natural-language instructions.

The current version is archived before saving the refined version, so you can revert if needed.

**Usage:**

```bash
mosaicx template refine <name> --instruction "..."
```

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--instruction` | TEXT | Yes | Natural-language refinement instruction |
| `--output` | PATH | No | Save refined template to a different path |

**Important:**
- Works with both built-in and user templates
- Refining a built-in template saves the result as a user template
- Previous versions are archived in `~/.mosaicx/templates/.history/`

**Examples:**

```bash
# Add a field using natural language
mosaicx template refine EchoReport \
  --instruction "add a field for tricuspid valve regurgitation severity"

# Remove fields
mosaicx template refine EchoReport \
  --instruction "remove wall_motion and add regional_wall_motion_abnormalities as a list"

# Make structural changes
mosaicx template refine CTReport \
  --instruction "add a LUNG-RADS category field as an integer 1-4"

# Save refined template to a custom location
mosaicx template refine chest_ct \
  --instruction "add fields for coronary calcification" \
  --output /path/to/custom_chest_ct.yaml
```

---

## `mosaicx template migrate`

Convert legacy JSON schemas from `~/.mosaicx/schemas/` to YAML templates in `~/.mosaicx/templates/`.

This is a one-time migration command for users upgrading from the old schema system to the unified template system.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--dry-run` | flag | No | Show what would be migrated without writing files |

**Examples:**

```bash
# Preview what would be migrated
mosaicx template migrate --dry-run

# Perform the migration
mosaicx template migrate
```

**What happens:**

1. Scans `~/.mosaicx/schemas/` for JSON schema files
2. Converts each to YAML template format
3. Saves to `~/.mosaicx/templates/{name}.yaml`
4. Skips any templates that already exist as YAML
5. Reports migrated, skipped, and errored files

---

## `mosaicx template history`

Show version history of a user template.

Every time you refine a template, the previous version is archived. This command lists all archived versions.

**Usage:**

```bash
mosaicx template history <name>
```

**Examples:**

```bash
mosaicx template history EchoReport
mosaicx template history CTLungNodule
```

**Output:**

Table showing:
- Version number (v1, v2, v3, ...)
- Date modified
- Current version

**Important:**
- Only user templates have version history (not built-in templates)
- History is stored in `~/.mosaicx/templates/.history/`

---

## `mosaicx template diff`

Compare the current version of a user template against a previous archived version.

**Usage:**

```bash
mosaicx template diff <name> --version <N>
```

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--version` | INT | Yes | Version number to compare against current |

**Examples:**

```bash
# Compare current EchoReport to version 2
mosaicx template diff EchoReport --version 2

# See what changed since version 1
mosaicx template diff CTReport --version 1
```

**Output:**

Shows:
- Added sections (green `+`)
- Removed sections (red `-`)
- Modified sections (yellow `~`) with details of what changed

---

## `mosaicx template revert`

Restore a user template to a previous version.

The current version is archived before reverting.

**Usage:**

```bash
mosaicx template revert <name> --version <N>
```

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--version` | INT | Yes | Version number to revert to |

**Examples:**

```bash
# Revert EchoReport to version 2
mosaicx template revert EchoReport --version 2

# Undo recent changes by reverting to version 1
mosaicx template revert CTReport --version 1
```

**What happens:**

1. Current template is archived as the next version number
2. Specified version becomes the current template
3. Confirmation message shows old and new version numbers

---

## `mosaicx template validate`

Validate a custom YAML template file.

Use this to check if your custom template is correctly formatted before using it with `mosaicx extract --template`.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--file` | PATH | Yes | Path to YAML template file to validate |

**Examples:**

```bash
# Validate a custom template
mosaicx template validate --file my_template.yaml

# Validate before using in extraction
mosaicx template validate --file chest_ct.yaml
```

**Output:**

If valid:
- Success message
- Model name
- List of fields

If invalid:
- Error message with details

---

## `mosaicx summarize`

Synthesize a patient timeline from multiple clinical reports.

Generates a narrative summary and extracts key events from one or more documents.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--document` | PATH | No* | Single document to summarize |
| `--dir` | PATH | No* | Directory of reports for one patient |
| `--patient` | TEXT | No | Patient identifier |
| `-o`, `--output` | PATH | No | Save output to JSON or YAML file |

**Important:**
- Must provide `--document` or `--dir`
- If using `--dir`, all TXT, MD, and MARKDOWN files will be loaded

**Examples:**

```bash
# Summarize a single document
mosaicx summarize --document clinic_note.pdf

# Summarize all reports in a directory
mosaicx summarize --dir ./patient_123_reports --patient "Patient 123"

# Single report with patient ID
mosaicx summarize --document discharge_summary.pdf --patient "John Doe"
```

**Output:**

Displays:
- Narrative summary (prose description of patient timeline)
- Timeline events table with columns: Date, Exam, Key Finding, Change from Prior

---

## `mosaicx deidentify`

Remove Protected Health Information (PHI) from clinical documents.

Supports three de-identification strategies:
1. **remove** (default): Replace PHI with `[REDACTED]`
2. **pseudonymize**: Replace PHI with fake but consistent values
3. **dateshift**: Shift dates by a random offset while preserving intervals

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--document` | PATH | No* | Single document to de-identify |
| `--dir` | PATH | No* | Directory of documents to de-identify |
| `--mode` | CHOICE | No | De-identification strategy: `remove`, `pseudonymize`, `dateshift` (default: `remove`) |
| `--regex-only` | flag | No | Use regex-only PHI scrubbing (no LLM call, faster) |
| `-o`, `--output` | PATH | No | Save output to JSON or YAML file (single document) |
| `--output-dir` | PATH | No | Directory for output files (batch mode) |
| `--format` | TEXT | No | Export format(s): `jsonl`, `parquet`, `csv` (can repeat) |
| `--workers` | INT | No | Number of parallel workers (default: 1) |
| `--resume` | flag | No | Resume from last checkpoint |

**Important:**
- Must provide `--document` or `--dir`
- Regex-only mode is faster but less accurate (only pattern matching)
- Full LLM mode is more thorough but slower and requires API calls

**Examples:**

```bash
# De-identify a single document (default: remove PHI)
mosaicx deidentify --document clinic_note.txt

# De-identify with pseudonymization
mosaicx deidentify --document report.txt --mode pseudonymize

# De-identify with date shifting
mosaicx deidentify --document discharge.txt --mode dateshift

# Batch de-identify a directory
mosaicx deidentify --dir ./reports --mode remove

# Fast regex-only mode (no LLM)
mosaicx deidentify --document report.txt --regex-only

# Parallel de-identification with 4 workers
mosaicx deidentify --dir ./patient_reports --workers 4 --mode pseudonymize

# Save single-document output to file
mosaicx deidentify --document clinic_note.txt -o deidentified.json

# Batch with output directory and export formats
mosaicx deidentify --dir ./reports --output-dir ./deidentified \
  --format jsonl --format csv

# Resume a failed batch
mosaicx deidentify --dir ./reports --output-dir ./deidentified --resume
```

**What gets redacted:**

- Patient names
- Medical record numbers (MRNs)
- Dates (birth dates, admission dates, etc.)
- Addresses
- Phone numbers
- Email addresses
- Other identifiers

**Output:**

Displays the de-identified text in a formatted panel. If processing a directory, shows output for each file.

---

## `mosaicx verify`

Verify an extraction or claim against a source document.

Checks whether structured extractions or free-text claims are supported by the original source document. Uses deterministic text analysis for the "quick" level (no LLM needed).

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--document` | PATH | No | Single source document (legacy single-source option) |
| `--sources` | PATH (repeatable) | No | One or more source documents to verify against |
| `--claim` | TEXT | No | A free-text claim to verify against the document |
| `--extraction` | PATH | No | JSON file with extraction output to verify |
| `--level` | CHOICE | No | Verification depth: `quick` (default), `standard`, `thorough` |
| `-o`, `--output` | PATH | No | Save verification result to JSON or YAML file |

**Important:**
- At least one of `--claim` or `--extraction` must be provided
- At least one of `--document` or `--sources` must be provided
- `quick` level uses deterministic checks (regex, text matching) -- no LLM needed, very fast
- `standard` level adds LLM spot-check of high-risk fields (measurements, severity, staging)
- `thorough` level runs a full LLM audit of all extracted fields
- Supported document formats: PDF, TXT, DOCX, MD, PNG, JPG, JPEG, TIF, TIFF

**Verdicts:**

| Verdict | Meaning |
|---------|---------|
| `verified` | All claims/fields are supported by the source text |
| `partially_supported` | Some fields supported, some could not be confirmed |
| `contradicted` | Source text contradicts the claim or extraction |
| `insufficient_evidence` | Source text does not contain enough information to judge |

**Examples:**

```bash
# Verify a free-text claim against a document
mosaicx verify --document ct_report.pdf --claim "2.3cm nodule in right upper lobe"

# Verify extraction output against the source document
mosaicx verify --document ct_report.pdf --extraction output.json

# Verify with thorough checking
mosaicx verify --document ct_report.pdf --extraction output.json --level thorough

# Save verification result to file
mosaicx verify --document ct_report.pdf --claim "normal chest CT" -o result.json
```

**Output:**

Displays:
- A decision-first adjudication block (`Decision`, `Requested`, `Effective`, fallback info)
- Claim mode: `Claim Comparison` with `Claimed`, `Source`, and `Evidence`
- Extraction mode: optional field-level mismatch table
- Machine-readable JSON/YAML includes `decision`, `support_score`, `verification_mode`, and fallback metadata

---

## `mosaicx query`

Query documents and data sources with natural language.

Load one or more data files and ask a question. Uses RLM (Recursive Language Model) -- the model writes and executes Python code in a sandboxed environment to answer your question.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--document` | PATH (repeatable) | No | Path to a data source (legacy alias) |
| `--sources` | TEXT (repeatable) | No | Paths, directories, or glob patterns (e.g. `"reports/*.txt"`) |
| `-q`, `--question` | TEXT | No | Ask one question and print answer with evidence |
| `--chat` | flag | No | Start a multi-turn query chat session |
| `--citations` | INT | No | Maximum citations returned per turn (default: `3`) |
| `-o`, `--output` | PATH | No | Save query turns/citations to JSON or YAML file |

**Important:**
- Requires [Deno](https://deno.land/) installed for the RLM code sandbox
- Requires a model with strong structured output capability (120B+ recommended)
- At least one `--document` or `--sources` input is required
- `-q` runs one-shot query; `--chat` runs multi-turn session with conversation memory
- Each answer includes evidence citations and grounding confidence
- If RLM is unavailable, query falls back to retrieval-only evidence mode

**Examples:**

```bash
# Ask a question about a CSV file
mosaicx query --document patient_data.csv -q "What is the mean age?"

# Query across multiple documents
mosaicx query --document data.csv --document notes.pdf -q "Summarize the key findings"

# Use glob-style source patterns
mosaicx query --sources "reports/*.txt" -q "List all pulmonary nodules with sizes"

# Multi-turn chat mode
mosaicx query --document report.pdf --chat

# Save the answer to a file
mosaicx query --document report.pdf -q "List all medications mentioned" -o answer.json

# Just load and inspect sources (no question)
mosaicx query --document data.csv --document results.json
```

**Output:**

Displays:
- Source catalog table (name, format, type, size)
- One-shot mode: answer + evidence citations + grounding confidence
- Chat mode: multi-turn answers with citations per turn

---

## `mosaicx optimize`

Optimize a DSPy pipeline using labeled examples.

Optimization uses progressive strategies (BootstrapFewShot -> MIPROv2 -> GEPA) to improve pipeline performance on your specific data.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--pipeline` | TEXT | No | Pipeline to optimize (e.g., `radiology`, `pathology`, `extract`) |
| `--trainset` | PATH | No | Training dataset in JSONL format |
| `--valset` | PATH | No | Validation dataset in JSONL format |
| `--budget` | CHOICE | No | Optimization budget: `light`, `medium`, `heavy` (default: `medium`) |
| `--save` | PATH | No | Custom save path for optimized program |
| `--list-pipelines` | flag | No | List available pipelines and exit |

**Budget presets:**

| Budget | Strategy | Cost | Time | Min Examples |
|--------|----------|------|------|--------------|
| `light` | BootstrapFewShot | ~$0.50 | ~5 min | 10 |
| `medium` | MIPROv2 | ~$3 | ~20 min | 10 |
| `heavy` | GEPA | ~$10 | ~45 min | 10 |

**Important:**
- Requires labeled training data in JSONL format
- Optimized programs are saved to `~/.mosaicx/optimized/` by default
- Use optimized programs with `mosaicx extract --optimized` or `mosaicx eval --optimized`

**Examples:**

```bash
# List available pipelines
mosaicx optimize --list-pipelines

# Light optimization (BootstrapFewShot)
mosaicx optimize --pipeline radiology \
  --trainset train.jsonl --budget light

# Medium optimization (MIPROv2, recommended)
mosaicx optimize --pipeline radiology \
  --trainset train.jsonl --valset val.jsonl --budget medium

# Heavy optimization (GEPA, best results)
mosaicx optimize --pipeline pathology \
  --trainset train.jsonl --valset val.jsonl --budget heavy

# Custom save location
mosaicx optimize --pipeline extract \
  --trainset examples.jsonl --budget medium \
  --save /path/to/optimized/custom_extractor.json

# Optimize the schema generator
mosaicx optimize --pipeline schema \
  --trainset schema_examples.jsonl --budget light
```

**Available pipelines:**

- `radiology` -- RadiologyReportStructurer
- `pathology` -- PathologyReportStructurer
- `extract` -- DocumentExtractor
- `summarize` -- ReportSummarizer
- `deidentify` -- Deidentifier
- `schema` -- SchemaGenerator

**Training data format (JSONL):**

Each line is a JSON object with inputs and expected outputs. Example for radiology:

```jsonl
{"report_text": "CT CHEST WITH CONTRAST...", "report_header": "CT CHEST", "expected": {...}}
{"report_text": "MRI BRAIN WITHOUT CONTRAST...", "report_header": "MRI BRAIN", "expected": {...}}
```

**Output:**

Displays:
- Optimization configuration
- Progressive strategy stages
- Training and validation scores
- Save path for optimized program

---

## `mosaicx eval`

Evaluate a pipeline against a labeled test set.

Runs the pipeline on each example in the test set and computes metrics.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--pipeline` | TEXT | Yes | Pipeline to evaluate (e.g., `radiology`, `pathology`) |
| `--testset` | PATH | Yes | Test dataset in JSONL format |
| `--optimized` | PATH | No | Path to optimized program (if not provided, uses baseline) |
| `--output` | PATH | No | Save detailed results as JSON |

**Examples:**

```bash
# Evaluate baseline (unoptimized) radiology pipeline
mosaicx eval --pipeline radiology --testset test.jsonl

# Evaluate optimized radiology pipeline
mosaicx eval --pipeline radiology --testset test.jsonl \
  --optimized ~/.mosaicx/optimized/radiology_optimized.json

# Save detailed results
mosaicx eval --pipeline pathology --testset test.jsonl \
  --optimized pathology_opt.json --output eval_results.json

# Evaluate the document extractor
mosaicx eval --pipeline extract --testset extract_test.jsonl
```

**Output:**

Displays:
- Evaluation configuration (pipeline, test set, examples count)
- Statistics table:
  - Count
  - Mean score
  - Median score
  - Standard deviation
  - Min/Max scores
- Score distribution histogram (0.0-0.2, 0.2-0.4, etc.)
- Detailed results (if `--output` specified)

**Test data format:**

Same JSONL format as training data. See `mosaicx optimize` for details.

---

## `mosaicx pipeline new`

Scaffold a new extraction pipeline from a built-in template.

Generates a complete DSPy pipeline module with lazy loading, mode registration, and a single-step extraction chain. The generated file follows the same pattern as the built-in radiology and pathology pipelines.

**Usage:**

```bash
mosaicx pipeline new <name> [--description "..."]
```

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `name` | TEXT | Yes | Pipeline name (auto-normalized to snake_case) |
| `-d`, `--description` | TEXT | No | One-line description of the pipeline |

**Examples:**

```bash
# Scaffold a cardiology pipeline
mosaicx pipeline new cardiology --description "Cardiology report structurer"

# PascalCase and kebab-case are normalized automatically
mosaicx pipeline new echo-report -d "Echocardiography report extraction"

# Minimal -- auto-generates a description
mosaicx pipeline new dermatology
```

**What gets generated:**

A new file at `mosaicx/pipelines/<name>.py` containing:
- Mode registration (so `--mode <name>` works with `mosaicx extract`)
- A DSPy Signature class for input/output fields
- A DSPy Module class with a `forward()` method
- Lazy loading boilerplate (module imports DSPy only when needed)

**After scaffolding:**

The command prints a wiring checklist of manual steps to complete the pipeline registration (adding to mode modules, evaluation registries, and CLI imports).

---

## `mosaicx mcp serve`

Start the MOSAICX Model Context Protocol (MCP) server.

The MCP server exposes MOSAICX tools (extract, verify, query, deidentify, schema generate, list schemas, list modes) for AI agents like Claude Code, Claude Desktop, and other MCP-compatible clients.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--transport` | CHOICE | No | Transport protocol: `stdio` or `sse` (default: `stdio`) |
| `--port` | INT | No | Port for the SSE HTTP server (default: `8080`) |

**Examples:**

```bash
# Start with stdio transport (default -- for Claude Code / Claude Desktop)
mosaicx mcp serve

# Start with SSE transport on port 9000
mosaicx mcp serve --transport sse --port 9000
```

**Important:**
- Requires the `mcp` optional dependency: `pip install mosaicx[mcp]`
- Use `stdio` transport for local integrations (Claude Code, Claude Desktop)
- Use `sse` transport for remote/network integrations

See the [MCP Server guide](mcp-server.md) for setup instructions with Claude Code and Claude Desktop.

---

## `mosaicx config show`

Print current configuration values.

Displays all MOSAICX settings, including:
- Language models (LM)
- Processing settings
- OCR settings
- Export settings
- Paths

**Examples:**

```bash
mosaicx config show
```

**Output sections:**

1. **Language Models**
   - `lm` -- Main language model
   - `lm_cheap` -- Cheaper model for simple tasks
   - `api_base` -- API base URL
   - `api_key` -- Masked API key

2. **Processing**
   - `default_template` -- Default template name
   - `completeness_threshold` -- Minimum completeness score (0-1)
   - `batch_workers` -- Default parallel workers
   - `checkpoint_every` -- Checkpoint frequency

3. **Document OCR**
   - `ocr_engine` -- OCR engine (`both`, `surya`, `chandra`)
   - `chandra_backend` -- Chandra backend (`vllm`, `hf`, `auto`)
   - `chandra_server_url` -- Chandra server URL (if applicable)
   - `quality_threshold` -- Minimum OCR quality (0-1)
   - `ocr_page_timeout` -- Timeout per page (seconds)
   - `force_ocr` -- Always use OCR (even for text PDFs)
   - `ocr_langs` -- OCR languages

4. **Export & Privacy**
   - `export_formats` -- Default export formats
   - `deidentify_mode` -- Default de-identification mode

5. **Paths**
   - `home_dir` -- MOSAICX home directory (`~/.mosaicx`)
   - `schema_dir` -- Schema directory
   - `optimized_dir` -- Optimized programs directory
   - `checkpoint_dir` -- Checkpoint directory
   - `log_dir` -- Log directory

---

## `mosaicx config set`

Set a configuration value (runtime only).

**Usage:**

```bash
mosaicx config set <key> <value>
```

**Important:**
- Changes are **not persisted** across sessions
- For permanent changes, use environment variables (`MOSAICX_*`) or a `.env` file

**Examples:**

```bash
# Set the main language model (runtime only)
mosaicx config set lm "openai/gpt-4"

# Set API base (runtime only)
mosaicx config set api_base "http://localhost:8000/v1"
```

**Recommended approach for persistent config:**

Create a `.env` file in your project directory or set environment variables:

```bash
# .env file
MOSAICX_LM=openai/gpt-4
MOSAICX_API_KEY=your-api-key-here
MOSAICX_API_BASE=http://localhost:11434/v1
MOSAICX_OCR_ENGINE=both
MOSAICX_BATCH_WORKERS=4
```

Or use environment variables:

```bash
export MOSAICX_LM="openai/gpt-4"
export MOSAICX_API_KEY="your-api-key-here"
export MOSAICX_API_BASE="http://localhost:11434/v1"
```

---

## Environment Variables

All configuration options can be set via environment variables with the `MOSAICX_` prefix.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MOSAICX_LM` | string | `openai/gpt-oss:120b` | Main language model |
| `MOSAICX_LM_CHEAP` | string | `openai/gpt-oss:20b` | Cheaper model for simple tasks |
| `MOSAICX_API_KEY` | string | `ollama` | API key |
| `MOSAICX_API_BASE` | string | `http://localhost:11434/v1` | API base URL |
| `MOSAICX_DEFAULT_TEMPLATE` | string | `auto` | Default template name |
| `MOSAICX_COMPLETENESS_THRESHOLD` | float | `0.7` | Minimum completeness score (0-1) |
| `MOSAICX_BATCH_WORKERS` | int | `1` | Number of parallel workers |
| `MOSAICX_CHECKPOINT_EVERY` | int | `50` | Checkpoint frequency |
| `MOSAICX_HOME_DIR` | path | `~/.mosaicx` | MOSAICX home directory |
| `MOSAICX_DEIDENTIFY_MODE` | choice | `remove` | De-identification mode (`remove`, `pseudonymize`, `dateshift`) |
| `MOSAICX_DEFAULT_EXPORT_FORMATS` | list | `["parquet", "jsonl"]` | Default export formats |
| `MOSAICX_OCR_ENGINE` | choice | `both` | OCR engine (`both`, `surya`, `chandra`) |
| `MOSAICX_CHANDRA_BACKEND` | choice | `auto` | Chandra backend (`vllm`, `hf`, `auto`) |
| `MOSAICX_CHANDRA_SERVER_URL` | string | `""` | Chandra server URL |
| `MOSAICX_QUALITY_THRESHOLD` | float | `0.6` | Minimum OCR quality (0-1) |
| `MOSAICX_OCR_PAGE_TIMEOUT` | int | `60` | OCR timeout per page (seconds) |
| `MOSAICX_FORCE_OCR` | bool | `false` | Always use OCR (even for text PDFs) |
| `MOSAICX_OCR_LANGS` | list | `["en", "de"]` | OCR languages (JSON array) |

**Examples:**

```bash
# Use GPT-4 via OpenAI API
export MOSAICX_LM="openai/gpt-4o"
export MOSAICX_API_KEY="sk-..."
export MOSAICX_API_BASE="https://api.openai.com/v1"

# Use a local vLLM server
export MOSAICX_LM="local/qwen-32b"
export MOSAICX_API_BASE="http://localhost:8000/v1"
export MOSAICX_API_KEY="none"

# Increase batch parallelism
export MOSAICX_BATCH_WORKERS=8

# Force OCR on all PDFs
export MOSAICX_FORCE_OCR=true

# Add Spanish to OCR languages
export MOSAICX_OCR_LANGS='["en", "de", "es"]'
```

---

## Common Workflows

### Extract data from a single radiology report

```bash
mosaicx extract --document ct_chest.pdf --mode radiology -o output.json
```

### Extract using a built-in template with completeness scoring

```bash
mosaicx extract --document ct_chest.pdf --template chest_ct --score -o output.json
```

### Batch process 100 pathology reports with 4 workers

```bash
mosaicx extract --dir ./biopsies --output-dir ./structured \
  --mode pathology --workers 4 --format jsonl --format parquet
```

### Create a custom template and use it

```bash
# Generate template from description
mosaicx template create \
  --describe "echo report with LVEF, valve grades, and wall motion"

# Use the template (auto-named by LLM, e.g., "EchoReport")
mosaicx extract --document echo.pdf --template EchoReport -o result.json
```

### Migrate legacy schemas to templates

```bash
# Preview what would be migrated
mosaicx template migrate --dry-run

# Perform the migration
mosaicx template migrate

# Use a migrated template
mosaicx extract --document echo.pdf --template EchoReport
```

### Optimize a pipeline and evaluate it

```bash
# Optimize
mosaicx optimize --pipeline radiology \
  --trainset train.jsonl --valset val.jsonl --budget medium

# Evaluate optimized version
mosaicx eval --pipeline radiology --testset test.jsonl \
  --optimized ~/.mosaicx/optimized/radiology_optimized.json
```

### De-identify a batch of clinic notes

```bash
mosaicx deidentify --dir ./clinic_notes --mode remove --workers 4
```

### Summarize a patient's longitudinal records

```bash
mosaicx summarize --dir ./patient_123 --patient "Patient 123"
```

### Extract and verify the output

```bash
# Extract structured data from a report
mosaicx extract --document ct_chest.pdf --template chest_ct -o output.json

# Verify the extraction against the source document
mosaicx verify --document ct_chest.pdf --extraction output.json
```

### Extract and query for follow-up analysis

```bash
# Extract structured data and save to JSON
mosaicx extract --document ct_chest.pdf --mode radiology -o structured.json

# Query the extracted data for specific findings
mosaicx query --document structured.json -q "Are there any critical findings?"

# Query across the source document and extraction together
mosaicx query --document ct_chest.pdf --document structured.json \
  -q "Summarize the nodule measurements"
```

---

## File Locations

MOSAICX stores data in `~/.mosaicx/` by default:

```
~/.mosaicx/
├── templates/            # User-created YAML templates
│   ├── EchoReport.yaml
│   ├── CTReport.yaml
│   └── .history/         # Archived template versions
│       ├── EchoReport_v1.yaml
│       └── EchoReport_v2.yaml
├── schemas/              # Legacy saved schemas (JSON)
│   ├── EchoReport.json
│   └── CTReport.json
├── optimized/            # Optimized DSPy programs
│   ├── radiology_optimized.json
│   └── pathology_optimized.json
├── checkpoints/          # Batch processing checkpoints
│   └── resume.json
└── logs/                 # Log files (future)
```

You can override the home directory with:

```bash
export MOSAICX_HOME_DIR=/path/to/custom/dir
```

---

## Tips for Beginners

1. **Start with auto mode**: Run `mosaicx extract --document report.pdf` to see what MOSAICX can do without any configuration.

2. **Use built-in modes**: For radiology and pathology reports, use `--mode radiology` or `--mode pathology` for best results.

3. **Try built-in templates**: Run `mosaicx template list` to see pre-defined templates for common exam types.

4. **Save your output**: Always use `-o output.json` to save the full structured data. Terminal output is summarized.

5. **Check available modes**: Run `mosaicx extract --list-modes` to see what's available.

6. **Create templates for repeated use**: If you process the same report type often, create a template with `mosaicx template create` and reuse it.

7. **Use batch mode for large datasets**: Don't run `extract` 100 times manually -- use `mosaicx extract --dir` with `--workers` for parallelism.

8. **Optimize for your data**: If you have labeled examples, use `mosaicx optimize` to improve accuracy on your specific reports.

9. **Resume failed batches**: If a batch crashes, use `--resume` to pick up where you left off.

10. **Migrate legacy schemas**: If you have JSON schemas from an older version, run `mosaicx template migrate` to convert them to YAML templates.

11. **Check your config**: Run `mosaicx config show` to see what models and settings you're using.

12. **Use environment variables**: Create a `.env` file with `MOSAICX_*` variables to avoid typing API keys and settings repeatedly.

---

## Troubleshooting

### "No API key configured"

Set your API key:
```bash
export MOSAICX_API_KEY="your-api-key-here"
```

Or add to `.env`:
```
MOSAICX_API_KEY=your-api-key-here
```

### "Document is empty"

- Check if the PDF is scanned (image-based) -- MOSAICX will use OCR automatically
- If OCR fails, try `--force-ocr` or adjust `MOSAICX_OCR_ENGINE`

### "Low OCR quality detected"

- The document is low-resolution or poorly scanned
- Results may be unreliable -- check the extracted text
- Try adjusting `MOSAICX_QUALITY_THRESHOLD` (lower = more permissive)

### "Template not found"

- Check available templates: `mosaicx template list`
- Verify the name matches exactly (case-sensitive)
- Ensure the template exists in `~/.mosaicx/templates/` or as a built-in
- For legacy schemas, the template resolution chain also checks `~/.mosaicx/schemas/`

### Batch processing is slow

- Increase workers: `--workers 4` or `--workers 8`
- Check if OCR is the bottleneck (try `--workers 1` and monitor CPU/GPU)
- For cloud LLMs, ensure your API has high rate limits

### Optimization fails with "not enough examples"

- You need at least 10 training examples
- See the "Min Examples" column in `mosaicx optimize --help`

---

## Getting Help

- **Command help**: `mosaicx <command> --help`
- **List modes**: `mosaicx extract --list-modes`
- **List templates**: `mosaicx template list`
- **List pipelines**: `mosaicx optimize --list-pipelines`
- **Show config**: `mosaicx config show`
- **Check version**: `mosaicx --version`

---

**End of CLI Reference**

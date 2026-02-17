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
1. **Auto mode** (no schema flag): LLM automatically determines what to extract
2. **Schema mode** (`--schema`): Use a saved schema from `~/.mosaicx/schemas/`
3. **Mode mode** (`--mode`): Use a built-in multi-step pipeline (radiology, pathology)
4. **Template mode** (`--template`): Use a custom YAML template file

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--document` | PATH | Yes | Path to the document (PDF, TXT, DOCX, PNG, JPG, TIFF) |
| `--schema` | TEXT | No | Name of a saved schema from `~/.mosaicx/schemas/` |
| `--mode` | TEXT | No | Extraction mode (e.g., `radiology`, `pathology`) |
| `--template` | PATH | No | Path to a YAML template file |
| `--optimized` | PATH | No | Path to an optimized DSPy program (`.json` file) |
| `-o`, `--output` | PATH | No | Save output to JSON or YAML file |
| `--list-modes` | flag | No | List available extraction modes and exit |

**Important:**
- `--schema`, `--mode`, and `--template` are mutually exclusive — use only one
- If none are provided, auto mode is used
- Supported formats: PDF, TXT, DOCX, MD, PNG, JPG, JPEG, TIF, TIFF

**Examples:**

```bash
# Auto mode — LLM decides what to extract from the document
mosaicx extract --document report.pdf

# List available modes
mosaicx extract --list-modes

# Radiology mode — 5-step pipeline for radiology reports
# Steps: classify exam → parse sections → extract technique → findings → impression
mosaicx extract --document ct_chest.pdf --mode radiology

# Pathology mode — 5-step pipeline for pathology reports
# Steps: classify specimen → parse sections → specimen details → findings → diagnosis
mosaicx extract --document biopsy.pdf --mode pathology

# Use a saved schema (must exist in ~/.mosaicx/schemas/)
mosaicx extract --document echo.pdf --schema EchoReport

# Use a custom YAML template
mosaicx extract --document report.pdf --template my_template.yaml

# Save output to JSON
mosaicx extract --document report.pdf --mode radiology -o output.json

# Save output to YAML
mosaicx extract --document report.pdf --mode radiology -o output.yaml

# Use an optimized program (from mosaicx optimize)
mosaicx extract --document report.pdf --mode radiology \
  --optimized ~/.mosaicx/optimized/radiology_optimized.json

# Combine mode with custom save location
mosaicx extract --document ct_report.pdf --mode radiology \
  -o /path/to/results/structured_report.json
```

**What you'll see:**

Without `--output`, results are displayed in the terminal as formatted tables. Use `--output` to save the full structured data as JSON or YAML.

---

## `mosaicx batch`

Batch-process a directory of documents using parallel workers.

All documents in the input directory are processed and saved as individual JSON files in the output directory. Optionally export consolidated formats (JSONL, Parquet).

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--input-dir` | PATH | Yes | Directory containing input documents |
| `--output-dir` | PATH | Yes | Directory for output files |
| `--schema` | TEXT | No | Schema name to use for all documents |
| `--mode` | TEXT | No | Extraction mode for all documents (e.g., `radiology`) |
| `--format` | TEXT | No | Output format(s): `jsonl`, `parquet`, `csv` (can repeat) |
| `--workers` | INT | No | Number of parallel workers (default: 1) |
| `--resume` | flag | No | Resume from last checkpoint |

**Important:**
- Each document produces a separate JSON file named `{original_filename}.json`
- Checkpoints are saved in `{output_dir}/.checkpoints/` if `--resume` is used
- Supported input formats: PDF, TXT, DOCX, MD, PNG, JPG, JPEG, TIF, TIFF
- `--format` can be specified multiple times (e.g., `--format jsonl --format parquet`)

**Examples:**

```bash
# Basic batch — auto mode, single worker
mosaicx batch --input-dir ./reports --output-dir ./structured

# Batch with radiology mode
mosaicx batch --input-dir ./ct_scans --output-dir ./structured_ct --mode radiology

# Batch with pathology mode and 4 parallel workers
mosaicx batch --input-dir ./biopsies --output-dir ./structured_path \
  --mode pathology --workers 4

# Use a saved schema
mosaicx batch --input-dir ./echo_reports --output-dir ./structured_echo \
  --schema EchoReport

# Export as JSONL (one JSON object per line)
mosaicx batch --input-dir ./reports --output-dir ./out \
  --mode radiology --format jsonl

# Export as both JSONL and Parquet
mosaicx batch --input-dir ./reports --output-dir ./out \
  --mode radiology --format jsonl --format parquet

# Resume a failed batch run
mosaicx batch --input-dir ./reports --output-dir ./out \
  --mode radiology --resume

# Maximum parallelism with 8 workers
mosaicx batch --input-dir ./large_dataset --output-dir ./processed \
  --mode radiology --workers 8 --format parquet
```

**Output structure:**

```
output_dir/
├── report1.json
├── report2.json
├── report3.json
├── results.jsonl          # if --format jsonl
├── results.parquet        # if --format parquet
└── .checkpoints/
    └── resume.json        # if --resume
```

**Resume behavior:**

If a batch crashes or is interrupted, use `--resume` to skip already-processed documents. MOSAICX will:
1. Load the checkpoint file
2. Skip documents already in the checkpoint
3. Process only remaining documents
4. Update the checkpoint every 50 documents (configurable via `MOSAICX_CHECKPOINT_EVERY`)

---

## `mosaicx schema generate`

Generate a Pydantic schema from natural language or a sample document.

Schemas are saved to `~/.mosaicx/schemas/` by default and can be reused with `mosaicx extract --schema`.

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--description` | TEXT | No* | Plain-English description of desired fields |
| `--from-document` | PATH | No* | Infer schema from a sample document |
| `--name` | TEXT | No | Override the schema class name |
| `--example-text` | TEXT | No | Example text for grounding the schema |
| `--output` | PATH | No | Custom save path (default: `~/.mosaicx/schemas/`) |

**Important:**
- Must provide `--description` or `--from-document` (or both)
- Schemas are saved as JSON files in `~/.mosaicx/schemas/{name}.json`

**Examples:**

```bash
# Generate from description
mosaicx schema generate \
  --description "echo report with LVEF, valve grades, chamber dimensions, and impression"

# Generate from sample document
mosaicx schema generate --from-document sample_echo.pdf

# Combine description and document
mosaicx schema generate \
  --description "extract vital signs and lab values" \
  --from-document clinic_note.pdf

# Override the auto-generated class name
mosaicx schema generate \
  --description "CT lung nodule report with LUNG-RADS score" \
  --name CTLungNodule

# Add example text for better grounding
mosaicx schema generate \
  --description "MRI brain with lesion measurements" \
  --example-text "Multiple T2 hyperintense foci in subcortical white matter..."

# Save to custom location
mosaicx schema generate \
  --description "chest x-ray findings" \
  --output /path/to/my_schemas/chest_xr.json
```

**What happens:**

1. LLM analyzes your description and/or document
2. Generates a Pydantic schema with appropriate field names, types, and descriptions
3. Saves the schema to `~/.mosaicx/schemas/{name}.json`
4. Displays the generated schema in the terminal

You can now use the schema with:
```bash
mosaicx extract --document new_echo.pdf --schema EchoReport
```

---

## `mosaicx schema list`

List all saved schemas in `~/.mosaicx/schemas/`.

**Examples:**

```bash
mosaicx schema list
```

**Output:**

Shows a table with:
- Schema name
- Number of fields
- Description (if available)

---

## `mosaicx schema show`

Display fields and types of a saved schema.

**Usage:**

```bash
mosaicx schema show <schema_name>
```

**Examples:**

```bash
# Show the EchoReport schema
mosaicx schema show EchoReport

# Show a custom schema
mosaicx schema show CTLungNodule
```

**Output:**

Displays a table with:
- Field name
- Type (str, int, float, List, etc.)
- Required (yes/no)
- Description

---

## `mosaicx schema refine`

Modify an existing schema using natural language or explicit operations.

You can refine schemas in two ways:
1. **LLM-driven** (`--instruction`): Natural language changes
2. **Manual** (`--add`, `--remove`, `--rename`): Direct field operations

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--schema` | TEXT | Yes | Name of the schema to refine |
| `--instruction` | TEXT | No | Natural-language refinement instruction |
| `--add` | TEXT | No | Add a field (`"field_name: type"`) |
| `--optional` | flag | No | Make the added field optional (use with `--add`) |
| `--description` | TEXT | No | Description for the added field (use with `--add`) |
| `--remove` | TEXT | No | Remove a field by name |
| `--rename` | TEXT | No | Rename a field (`"old_name=new_name"`) |

**Important:**
- Must provide one of: `--instruction`, `--add`, `--remove`, or `--rename`
- Changes are saved to the same schema file
- Original version is archived to `~/.mosaicx/schemas/.archive/`

**Examples:**

```bash
# LLM-driven refinement
mosaicx schema refine --schema EchoReport \
  --instruction "add a field for tricuspid valve regurgitation severity"

# Add a required field
mosaicx schema refine --schema CTReport \
  --add "lung_rads_score: int" \
  --description "LUNG-RADS category (1-4)"

# Add an optional field
mosaicx schema refine --schema PathReport \
  --add "her2_status: str" --optional \
  --description "HER2/neu status if tested"

# Remove a field
mosaicx schema refine --schema EchoReport --remove chamber_dimensions

# Rename a field
mosaicx schema refine --schema CTReport --rename "findings=radiology_findings"

# Multiple LLM-driven changes
mosaicx schema refine --schema EchoReport \
  --instruction "remove wall_motion and add regional_wall_motion_abnormalities as a list"
```

---

## `mosaicx schema history`

Show version history of a schema.

Every time you refine a schema, the previous version is archived. This command lists all archived versions.

**Usage:**

```bash
mosaicx schema history <schema_name>
```

**Examples:**

```bash
mosaicx schema history EchoReport
mosaicx schema history CTLungNodule
```

**Output:**

Table showing:
- Version number (v1, v2, v3, ...)
- Number of fields
- Date modified

---

## `mosaicx schema diff`

Compare current schema against a previous version.

**Usage:**

```bash
mosaicx schema diff <schema_name> --version <N>
```

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--version` | INT | Yes | Version number to compare against current |

**Examples:**

```bash
# Compare current EchoReport to version 2
mosaicx schema diff EchoReport --version 2

# See what changed since version 1
mosaicx schema diff PathReport --version 1
```

**Output:**

Shows:
- Added fields (green `+`)
- Removed fields (red `-`)
- Modified fields (yellow `~`) with details of what changed

---

## `mosaicx schema revert`

Restore a schema to a previous version.

The current version is archived before reverting.

**Usage:**

```bash
mosaicx schema revert <schema_name> --version <N>
```

**Options:**

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--version` | INT | Yes | Version number to revert to |

**Examples:**

```bash
# Revert EchoReport to version 2
mosaicx schema revert EchoReport --version 2

# Undo recent changes by reverting to version 1
mosaicx schema revert PathReport --version 1
```

**What happens:**

1. Current schema is archived as the next version number
2. Specified version becomes the current schema
3. Displays a diff showing what changed

---

## `mosaicx template list`

List built-in radiology report templates.

Templates are pre-defined YAML schemas for common radiology exams.

**Examples:**

```bash
mosaicx template list
```

**Output:**

Table showing:
- Template name
- Exam type (e.g., CT Chest, MRI Brain)
- RadReport ID (if applicable)
- Description

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
| `--format` | TEXT | No | Output format(s) (not yet implemented) |

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
- Number of extracted timeline events
- Event details (if any)

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
| `--workers` | INT | No | Number of parallel workers (default: 1) |

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

## `mosaicx optimize`

Optimize a DSPy pipeline using labeled examples.

Optimization uses progressive strategies (BootstrapFewShot → MIPROv2 → GEPA) to improve pipeline performance on your specific data.

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

- `radiology` — RadiologyReportStructurer
- `pathology` — PathologyReportStructurer
- `extract` — DocumentExtractor
- `summarize` — ReportSummarizer
- `deidentify` — Deidentifier
- `schema` — SchemaGenerator

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
   - `lm` — Main language model
   - `lm_cheap` — Cheaper model for simple tasks
   - `api_base` — API base URL
   - `api_key` — Masked API key

2. **Processing**
   - `default_template` — Default template name
   - `completeness_threshold` — Minimum completeness score (0-1)
   - `batch_workers` — Default parallel workers
   - `checkpoint_every` — Checkpoint frequency

3. **Document OCR**
   - `ocr_engine` — OCR engine (`both`, `surya`, `chandra`)
   - `chandra_backend` — Chandra backend (`vllm`, `hf`, `auto`)
   - `chandra_server_url` — Chandra server URL (if applicable)
   - `quality_threshold` — Minimum OCR quality (0-1)
   - `ocr_page_timeout` — Timeout per page (seconds)
   - `force_ocr` — Always use OCR (even for text PDFs)
   - `ocr_langs` — OCR languages

4. **Export & Privacy**
   - `export_formats` — Default export formats
   - `deidentify_mode` — Default de-identification mode

5. **Paths**
   - `home_dir` — MOSAICX home directory (`~/.mosaicx`)
   - `schema_dir` — Schema directory
   - `optimized_dir` — Optimized programs directory
   - `checkpoint_dir` — Checkpoint directory
   - `log_dir` — Log directory

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
| `MOSAICX_DEFAULT_EXPORT_FORMATS` | list | `["parquet", "jsonl"]` | Default export formats (JSON array) |
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

### Batch process 100 pathology reports with 4 workers

```bash
mosaicx batch --input-dir ./biopsies --output-dir ./structured \
  --mode pathology --workers 4 --format jsonl --format parquet
```

### Create a custom schema and use it

```bash
# Generate schema from description
mosaicx schema generate \
  --description "echo report with LVEF, valve grades, and wall motion"

# Use the schema (auto-named by LLM, e.g., "EchoReport")
mosaicx extract --document echo.pdf --schema EchoReport -o result.json
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

---

## File Locations

MOSAICX stores data in `~/.mosaicx/` by default:

```
~/.mosaicx/
├── schemas/              # Saved schemas (JSON)
│   ├── EchoReport.json
│   ├── CTReport.json
│   └── .archive/         # Archived schema versions
│       ├── EchoReport_v1.json
│       └── EchoReport_v2.json
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

3. **Save your output**: Always use `-o output.json` to save the full structured data. Terminal output is summarized.

4. **Check available modes**: Run `mosaicx extract --list-modes` to see what's available.

5. **Create schemas for repeated use**: If you process the same report type often, create a schema with `mosaicx schema generate` and reuse it.

6. **Use batch mode for large datasets**: Don't run `extract` 100 times manually — use `mosaicx batch` with `--workers` for parallelism.

7. **Optimize for your data**: If you have labeled examples, use `mosaicx optimize` to improve accuracy on your specific reports.

8. **Resume failed batches**: If a batch crashes, use `--resume` to pick up where you left off.

9. **Check your config**: Run `mosaicx config show` to see what models and settings you're using.

10. **Use environment variables**: Create a `.env` file with `MOSAICX_*` variables to avoid typing API keys and settings repeatedly.

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

- Check if the PDF is scanned (image-based) — MOSAICX will use OCR automatically
- If OCR fails, try `--force-ocr` or adjust `MOSAICX_OCR_ENGINE`

### "Low OCR quality detected"

- The document is low-resolution or poorly scanned
- Results may be unreliable — check the extracted text
- Try adjusting `MOSAICX_QUALITY_THRESHOLD` (lower = more permissive)

### "Schema not found"

- Check available schemas: `mosaicx schema list`
- Verify the name matches exactly (case-sensitive)
- Ensure the schema exists in `~/.mosaicx/schemas/`

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
- **List pipelines**: `mosaicx optimize --list-pipelines`
- **Show config**: `mosaicx config show`
- **Check version**: `mosaicx --version`

---

**End of CLI Reference**

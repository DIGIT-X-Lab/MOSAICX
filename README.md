<div align="center">
  <img src="assets/mosaicx_logo.png" alt="MOSAICX" width="800"/>
</div>

<p align="center">
  <a href="https://pypi.org/project/mosaicx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mosaicx.svg?label=PyPI&style=flat-square&logo=python&logoColor=white&color=bd93f9"></a>
  <a href="https://doi.org/10.5281/zenodo.17601890"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17601890-ff79c6?style=flat-square&labelColor=282a36&logo=zenodo&logoColor=white"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-50fa7b?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-ff79c6?style=flat-square&logo=apache&logoColor=white"></a>
  <a href="https://pepy.tech/project/mosaicx"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mosaicx?style=flat-square&color=8be9fd&label=Downloads"></a>
</p>

<p align="center">
  <strong><a href="https://www.linkedin.com/company/digitx-lmu/">DIGIT-X Lab</a> · LMU Munich</strong><br>
  Structure first. Insight follows.
</p>

---

Turn unstructured medical documents (radiology reports, clinical notes, pathology summaries) into validated, machine-readable JSON. Runs locally via Ollama — no PHI leaves your machine.

## Quick start

```bash
# 1. Install Ollama (skip if already installed)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve                    # start the daemon (runs on :11434)

# 2. Pull a model
ollama pull gpt-oss:20b         # fast  — 16 GB RAM
ollama pull gpt-oss:120b        # best  — 64 GB RAM

# 3. Install MOSAICX
pip install mosaicx              # or: uv add mosaicx / pipx install mosaicx

# 4. Verify
mosaicx --version
```

Defaults point to Ollama on localhost — no `.env` needed for local use. See [Configuration](#configuration) to customize.

## What it does

| Command | Purpose |
|---------|---------|
| `mosaicx extract` | Extract structured data (auto, schema, or mode) |
| `mosaicx batch` | Batch-process documents with schema or mode |
| `mosaicx schema generate` | Create a Pydantic schema from a plain-English description |
| `mosaicx schema list` | List saved schemas |
| `mosaicx schema show <name>` | Inspect a schema's fields and types |
| `mosaicx schema refine` | Edit a schema (LLM-driven or manual field ops) |
| `mosaicx schema history <name>` | Show version history of a schema |
| `mosaicx schema diff <name>` | Compare current schema against a previous version |
| `mosaicx schema revert <name>` | Restore a previous version of a schema |
| `mosaicx summarize` | Synthesize a patient timeline from multiple reports |
| `mosaicx deidentify` | Remove PHI (LLM + regex belt-and-suspenders) |
| `mosaicx template list` | List built-in radiology templates |
| `mosaicx template validate` | Validate a custom YAML template |
| `mosaicx optimize` | Tune a DSPy pipeline (BootstrapFewShot / MIPROv2) |
| `mosaicx config show` | Print current configuration |

Run any command with `--help` for full options.

## Usage examples

### Schema management

Schemas live in `~/.mosaicx/schemas/` as JSON files named after their class (e.g., `EchoReport.json`). Generate, inspect, and refine them entirely from the CLI.

**Generate** — describe what you need in plain English:

```bash
mosaicx schema generate \
  --description "echocardiography report with LVEF, valve grades, impression"

# optionally set the schema name (default: LLM-chosen):
mosaicx schema generate \
  --description "patient name, dob, age, sex, blood pressure" \
  --name PatientVitals
```

**List** saved schemas:

```bash
mosaicx schema list
```

**Refine** — add, remove, or rename fields, or let the LLM restructure:

```bash
# Add a required field (default)
mosaicx schema refine --schema EchoReport --add "rvsp: float"

# Add an optional field with a description
mosaicx schema refine --schema EchoReport \
  --add "hospital_name: str" \
  --optional \
  --description "Name of the treating hospital"

# Remove or rename fields
mosaicx schema refine --schema EchoReport --remove clinical_impression
mosaicx schema refine --schema EchoReport --rename "lvef=lvef_percent"

# LLM-driven refinement
mosaicx schema refine --schema EchoReport \
  --instruction "add a field for pericardial effusion severity as an enum"
```

Every generate and refine auto-archives the previous version. See [Version history](#version-history) for how to inspect and revert.

### Version history

Every time a schema is saved (via `generate` or `refine`), the previous version is archived to `~/.mosaicx/schemas/.history/`. Nothing is ever silently lost.

**List versions:**

```bash
mosaicx schema history EchoReport
# Version  Fields  Date
# v1       5       2026-02-15 10:30
# v2       6       2026-02-15 10:35
# current  6       2026-02-15 10:35
```

**Compare against a previous version:**

```bash
mosaicx schema diff EchoReport --version 1
# + rvsp           (float)
# ~ lvef_percent   type: int -> float
```

**Revert to a previous version:**

```bash
mosaicx schema revert EchoReport --version 1
# ✓ Reverted EchoReport to v1 (archived current as v3)
```

Reverting archives the current schema first, so you can always get back to it.

### Extraction

Extract structured data from clinical documents. Three modes:

**Auto mode** — LLM reads the document and figures out what to extract:

```bash
mosaicx extract --document report.pdf
```

**Schema mode** — extract into a user-defined schema:

```bash
# First, generate a schema
mosaicx schema generate --description "echo report with LVEF, valve grades, impression"

# Then extract into it
mosaicx extract --document report.pdf --schema EchoReport
```

**Specialized modes** — domain-specific multi-step pipelines:

```bash
# Radiology: 5-step chain with measurements, scoring (BI-RADS, Lung-RADS), anatomy codes
mosaicx extract --document ct_report.pdf --mode radiology

# Pathology: 5-step chain with histology, TNM staging, biomarkers
mosaicx extract --document pathology_report.pdf --mode pathology

# List available modes
mosaicx extract --list-modes
```

**YAML templates** — extract into a compiled YAML template:

```bash
mosaicx extract --document report.pdf --template ./custom_template.yaml
```

### Batch processing

```bash
# Auto mode (LLM infers schema for each document)
mosaicx batch --input-dir ./reports --output-dir ./structured

# With a schema
mosaicx batch --input-dir ./reports --output-dir ./structured --schema EchoReport

# With a mode
mosaicx batch --input-dir ./reports --output-dir ./structured --mode radiology

# Resume after interruption
mosaicx batch --input-dir ./reports --output-dir ./structured --mode radiology --resume
```

### Summarize reports

```bash
mosaicx summarize --dir ./patient_001/ --patient P001
mosaicx summarize --document single_report.pdf
```

### De-identify

```bash
mosaicx deidentify --document note.txt                  # LLM + regex
mosaicx deidentify --dir ./notes --regex-only           # regex only, no LLM
mosaicx deidentify --document note.txt --mode pseudonymize
```

## Configuration

All settings live under the `MOSAICX_` prefix. Set them as environment variables or in a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MOSAICX_LM` | `openai/gpt-oss:120b` | Primary LLM (litellm format) |
| `MOSAICX_LM_CHEAP` | `openai/gpt-oss:20b` | Fallback / cheap model |
| `MOSAICX_API_KEY` | `ollama` | API key (`ollama` for local Ollama) |
| `MOSAICX_API_BASE` | `http://localhost:11434/v1` | LLM endpoint URL |
| `MOSAICX_OCR_ENGINE` | `both` | `surya`, `chandra`, or `both` |
| `MOSAICX_FORCE_OCR` | `false` | Force OCR even on native PDFs |
| `MOSAICX_OCR_LANGS` | `en,de` | OCR language hints |
| `MOSAICX_BATCH_WORKERS` | `4` | Parallel workers for batch mode |
| `MOSAICX_COMPLETENESS_THRESHOLD` | `0.7` | Minimum extraction quality (0-1) |
| `MOSAICX_DEIDENTIFY_MODE` | `remove` | `remove`, `pseudonymize`, `dateshift` |
| `MOSAICX_DEFAULT_EXPORT_FORMATS` | `parquet,jsonl` | Batch export formats |

View the active config:

```bash
mosaicx config show
```

## LLM backends

MOSAICX talks to any OpenAI-compatible endpoint via DSPy + litellm. Defaults point to Ollama on localhost — override with env vars for other backends.

| Backend | Port | Example |
|---------|------|---------|
| **Ollama** | 11434 | Works out-of-the-box, no config needed |
| **llama.cpp** | 8080 | `llama-server -m model.gguf --port 8080` |
| **vLLM** | 8000 | `vllm serve meta-llama/Llama-3.1-70B-Instruct` |
| **SGLang** | 30000 | `python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct` |
| **vLLM-MLX** | 8000 | `vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit` (Apple Silicon) |

```bash
# Ollama (default — no env vars needed)
mosaicx schema generate --description "..."

# llama.cpp / vLLM on a remote GPU server (e.g., DGX Spark)
# 1. SSH tunnel the server port to localhost:
ssh -L 8080:localhost:8080 user@dgx-spark      # llama.cpp (port 8080)
ssh -L 8000:localhost:8000 user@dgx-spark      # vLLM      (port 8000)
ssh -L 30000:localhost:30000 user@dgx-spark    # SGLang    (port 30000)

# 2. Point MOSAICX at the forwarded port:
export MOSAICX_LM=openai/your-model
export MOSAICX_API_BASE=http://localhost:8080/v1    # llama.cpp
export MOSAICX_API_BASE=http://localhost:8000/v1    # vLLM
export MOSAICX_API_BASE=http://localhost:30000/v1   # SGLang

# OpenAI / Together AI / Groq
export MOSAICX_LM=openai/gpt-4o
export MOSAICX_API_KEY=sk-...
export MOSAICX_API_BASE=https://api.openai.com/v1
```

For vLLM and SGLang the model name must match what the server loaded. For llama.cpp any name works (only one model loaded at a time). The default `api_key` (`ollama`) is ignored by servers that don't check auth.

### Local inference on Apple Silicon (vLLM-MLX)

[vLLM-MLX](https://github.com/waybarrios/vllm-mlx) runs vLLM-compatible inference natively on Apple Silicon Macs using MLX. No remote GPU server needed — uses unified memory for quantized models up to ~70B on a 128 GB Mac.

```bash
# 1. Install
uv tool install git+https://github.com/waybarrios/vllm-mlx.git

# 2. Serve a quantized model (models from mlx-community on Hugging Face)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# With continuous batching for concurrent requests:
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching

# 3. Point MOSAICX at it
export MOSAICX_LM=openai/default
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy
```

> **Note:** vLLM-MLX uses `default` as the model name. Use `openai/default` for `MOSAICX_LM`.

### Batch processing on a GPU server (vLLM / SGLang)

For batch-processing many documents, a dedicated inference server (**vLLM** or **SGLang**) is recommended over Ollama. Both use continuous batching and advanced memory management to handle concurrent requests efficiently without duplicating VRAM per slot.

#### vLLM

**On the GPU server** (e.g., DGX Spark with 128 GB VRAM):

```bash
# Large model (120B) — constrain concurrency to avoid OOM
vllm serve gpt-oss:120b \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --port 8000

# Smaller model (20B) — higher concurrency, faster throughput
vllm serve gpt-oss:20b \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --port 8000
```

**On your Mac:**

```bash
# 1. SSH tunnel
ssh -L 8000:localhost:8000 user@dgx-spark

# 2. Point MOSAICX at vLLM
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy

# 3. Match --workers to --max-num-seqs on the server
mosaicx batch --input-dir ./reports --output-dir ./structured \
  --mode radiology --format jsonl --workers 4
```

#### SGLang

**On the GPU server** (e.g., DGX Spark with 128 GB VRAM):

```bash
# Large model (120B) — constrain concurrency to avoid OOM
python -m sglang.launch_server \
  --model-path gpt-oss:120b \
  --mem-fraction-static 0.90 \
  --max-running-requests 4 \
  --port 30000

# Smaller model (20B) — higher concurrency, faster throughput
python -m sglang.launch_server \
  --model-path gpt-oss:20b \
  --mem-fraction-static 0.90 \
  --max-running-requests 16 \
  --port 30000
```

**On your Mac:**

```bash
# 1. SSH tunnel
ssh -L 30000:localhost:30000 user@dgx-spark

# 2. Point MOSAICX at SGLang
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:30000/v1
export MOSAICX_API_KEY=dummy

# 3. Match --workers to --max-running-requests on the server
mosaicx batch --input-dir ./reports --output-dir ./structured \
  --mode radiology --format jsonl --workers 4
```

#### Tuning workers

| Setup | Server concurrency | `--workers` | Notes |
|-------|-------------------|-------------|-------|
| 120B, 128 GB VRAM | 4 | 4 | Best quality, moderate speed |
| 20B, 128 GB VRAM | 16 | 16 | Faster throughput, good quality |
| 120B quantized (AWQ) | 8 | 8 | Balance of quality and speed |

> **Tip:** Setting `--workers` higher than the server's max concurrency (`--max-num-seqs` for vLLM, `--max-running-requests` for SGLang) wastes overhead — requests queue on the server side. Keep them matched.

## OCR engines

MOSAICX ships with two OCR engines that run in parallel by default:

| Engine | Approach | Best for |
|--------|----------|----------|
| **Surya** | Layout detection + recognition | Clean printed text, fast |
| **Chandra** | Vision-Language Model (Qwen3-VL 9B) | Handwriting, complex layouts, tables |

The dual-engine pipeline scores both outputs per page and picks the best. Override with `MOSAICX_OCR_ENGINE=surya` or `chandra` if you only need one.

## Development

```bash
git clone https://github.com/LalithShiyam/MOSAICX.git
cd MOSAICX
pip install -e ".[dev]"          # or: uv sync --group dev
pytest tests/ -q                 # 255 tests
```

## Citation

```bibtex
@software{mosaicx2025,
  title   = {MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction},
  author  = {Sundar, Lalith Kumar Shiyam and DIGIT-X Lab},
  year    = {2025},
  url     = {https://github.com/LalithShiyam/MOSAICX},
  doi     = {10.5281/zenodo.17601890}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).

## Contact

- Research: [lalith.shiyam@med.uni-muenchen.de](mailto:lalith.shiyam@med.uni-muenchen.de)
- Commercial: [lalith@zenta.solutions](mailto:lalith@zenta.solutions)
- Issues: [github.com/LalithShiyam/MOSAICX/issues](https://github.com/LalithShiyam/MOSAICX/issues)

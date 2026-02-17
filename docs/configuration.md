# Configuration

This guide covers all MOSAICX configuration options and explains how to connect different LLM backends. Whether you're a beginner running Ollama locally or an advanced user connecting to remote GPU servers, this guide will walk you through everything step by step.

## How Configuration Works

MOSAICX uses environment variables with the `MOSAICX_` prefix to control behavior. These variables can be set in three ways:

1. **Environment variables** — exported in your shell session
2. **A `.env` file** — placed in the directory where you run MOSAICX
3. **Defaults** — MOSAICX ships with sensible defaults optimized for local Ollama

Configuration is resolved in this order (highest priority first):

1. Environment variables (e.g., `export MOSAICX_LM=...`)
2. `.env` file values
3. Built-in defaults

### View Your Current Configuration

To see what configuration MOSAICX is using right now, run:

```bash
mosaicx config show
```

This prints all active settings, including LLM endpoints, OCR engine choices, batch processing settings, and file paths.

### Defaults Work Out-of-the-Box

If you installed Ollama and are running it on `localhost:11434`, MOSAICX will work immediately without any configuration. The defaults are:

- **Primary LLM:** `openai/gpt-oss:120b`
- **Cheap LLM:** `openai/gpt-oss:20b`
- **API endpoint:** `http://localhost:11434/v1`
- **API key:** `ollama` (used as a placeholder for local Ollama)

You only need to configure MOSAICX if you are using a different LLM backend, changing OCR settings, or tuning batch processing behavior.

## Setting Up a `.env` File

A `.env` file is a plain text file that stores configuration in a simple `KEY=VALUE` format. MOSAICX automatically reads `.env` files in the current directory when commands are run.

### Creating a `.env` File

Create a file named `.env` in the directory where you will run MOSAICX:

```bash
# Navigate to your working directory
cd /path/to/your/project

# Create the .env file
touch .env
```

Open `.env` in any text editor and add your settings. Here is a complete example with all common settings:

```bash
# ========================================
# LLM Settings
# ========================================
# Primary LLM used for extraction, schema generation, etc.
MOSAICX_LM=openai/gpt-oss:120b

# Cheap/fast model for simple tasks (e.g., classification)
MOSAICX_LM_CHEAP=openai/gpt-oss:20b

# API key (use "ollama" for local Ollama, or your actual API key for cloud providers)
MOSAICX_API_KEY=ollama

# LLM API endpoint URL
MOSAICX_API_BASE=http://localhost:11434/v1

# ========================================
# Processing Settings
# ========================================
# Default extraction template (usually "auto" to let the LLM infer structure)
MOSAICX_DEFAULT_TEMPLATE=auto

# Minimum extraction quality score (0.0 to 1.0)
# Lower values accept lower-quality extractions, higher values enforce stricter quality
MOSAICX_COMPLETENESS_THRESHOLD=0.7

# Number of parallel workers for batch processing
# Start with 1-4, increase if you have more CPU cores and RAM
MOSAICX_BATCH_WORKERS=1

# Save a checkpoint every N documents during batch processing
# Useful for resuming large batch jobs after interruptions
MOSAICX_CHECKPOINT_EVERY=50

# ========================================
# OCR Settings
# ========================================
# OCR engine: "both" (runs Surya + Chandra in parallel), "surya", or "chandra"
MOSAICX_OCR_ENGINE=both

# Chandra backend: "vllm" (fast, GPU-only), "hf" (CPU-compatible), or "auto"
MOSAICX_CHANDRA_BACKEND=auto

# Remote Chandra server URL (leave empty for local inference)
MOSAICX_CHANDRA_SERVER_URL=

# Minimum OCR quality score per page (0.0 to 1.0)
# Pages below this threshold trigger a warning
MOSAICX_QUALITY_THRESHOLD=0.6

# OCR timeout per page in seconds
# Increase for large/complex pages, decrease for faster failure on problematic pages
MOSAICX_OCR_PAGE_TIMEOUT=60

# Force OCR even on native PDFs (PDFs with embedded text)
# Set to "true" if you want to OCR all PDFs regardless of text layer
MOSAICX_FORCE_OCR=false

# OCR language hints (comma-separated ISO 639-1 codes)
# Improves accuracy when documents contain non-English text
MOSAICX_OCR_LANGS=en,de

# ========================================
# Privacy and Export
# ========================================
# Default de-identification mode: "remove", "pseudonymize", or "dateshift"
MOSAICX_DEIDENTIFY_MODE=remove

# Default export formats for batch processing (comma-separated)
# Options: parquet, jsonl, csv
MOSAICX_DEFAULT_EXPORT_FORMATS=parquet,jsonl

# ========================================
# Paths
# ========================================
# Home directory for schemas, optimized programs, checkpoints, and logs
# Default: ~/.mosaicx
MOSAICX_HOME_DIR=/Users/yourusername/.mosaicx
```

### Using the `.env` File

Once you have created a `.env` file, MOSAICX automatically reads it when you run commands in that directory:

```bash
cd /path/to/your/project
mosaicx extract --document report.pdf
```

No need to manually export variables — MOSAICX handles it for you.

### Alternative: Export Environment Variables

Instead of using a `.env` file, you can export variables directly in your shell:

```bash
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_BATCH_WORKERS=8
```

These exports last for the current shell session. If you close the terminal, you'll need to export them again.

To make exports permanent, add them to your shell configuration file:

- **Bash:** `~/.bashrc` or `~/.bash_profile`
- **Zsh:** `~/.zshrc`
- **Fish:** `~/.config/fish/config.fish`

## All Configuration Variables

Below is a complete reference of every MOSAICX configuration variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `MOSAICX_LM` | `openai/gpt-oss:120b` | Primary LLM in litellm format (e.g., `openai/model-name`, `anthropic/claude-3-opus`) |
| `MOSAICX_LM_CHEAP` | `openai/gpt-oss:20b` | Cheap/fast model for simple tasks like classification and section parsing |
| `MOSAICX_API_KEY` | `ollama` | API key for the LLM provider. Use `ollama` for local Ollama servers, or your actual key for cloud APIs (e.g., `sk-...` for OpenAI) |
| `MOSAICX_API_BASE` | `http://localhost:11434/v1` | Base URL for the LLM API endpoint. Must end with `/v1` for OpenAI-compatible servers |
| `MOSAICX_DEFAULT_TEMPLATE` | `auto` | Default extraction template name. `auto` lets the LLM infer structure |
| `MOSAICX_COMPLETENESS_THRESHOLD` | `0.7` | Minimum extraction quality score (0.0 to 1.0). Higher values enforce stricter extraction quality |
| `MOSAICX_BATCH_WORKERS` | `1` | Number of parallel workers for batch processing. Increase for faster processing if you have sufficient RAM and CPU cores |
| `MOSAICX_CHECKPOINT_EVERY` | `50` | Save a checkpoint every N documents in batch mode. Useful for resuming interrupted batch jobs |
| `MOSAICX_HOME_DIR` | `~/.mosaicx` | Home directory for storing schemas, optimized programs, checkpoints, and logs |
| `MOSAICX_DEIDENTIFY_MODE` | `remove` | Default de-identification strategy. Options: `remove` (strip PHI), `pseudonymize` (replace with fake data), `dateshift` (shift dates consistently) |
| `MOSAICX_DEFAULT_EXPORT_FORMATS` | `parquet,jsonl` | Default export formats for batch processing (comma-separated). Options: `parquet`, `jsonl` |
| `MOSAICX_OCR_ENGINE` | `both` | OCR engine to use. Options: `both` (Surya + Chandra in parallel), `surya` (traditional OCR), `chandra` (VLM-based OCR) |
| `MOSAICX_CHANDRA_BACKEND` | `auto` | Backend for Chandra OCR. Options: `vllm` (fast, GPU-only), `hf` (Hugging Face Transformers, CPU-compatible), `auto` (auto-detect) |
| `MOSAICX_CHANDRA_SERVER_URL` | (empty) | URL for a remote Chandra server. Leave empty for local inference |
| `MOSAICX_QUALITY_THRESHOLD` | `0.6` | Minimum OCR quality score per page (0.0 to 1.0). Pages below this threshold trigger a warning |
| `MOSAICX_OCR_PAGE_TIMEOUT` | `60` | OCR timeout per page in seconds. Increase for complex pages, decrease for faster failure on problematic pages |
| `MOSAICX_FORCE_OCR` | `false` | Force OCR even on native PDFs with embedded text. Set to `true` to OCR all PDFs regardless of text layer |
| `MOSAICX_OCR_LANGS` | `en,de` | OCR language hints as comma-separated ISO 639-1 codes (e.g., `en,de,fr,es`). Improves accuracy for non-English text |

## Directory Structure

MOSAICX stores schemas, optimized programs, checkpoints, and logs in `~/.mosaicx/` by default. You can change this with `MOSAICX_HOME_DIR`.

Here is what the directory structure looks like:

```
~/.mosaicx/
├── schemas/                # Saved Pydantic schemas (.json files)
│   ├── EchoReport.json     # Example schema
│   ├── PatientVitals.json  # Another schema
│   └── .history/           # Auto-archived previous versions of schemas
│       ├── EchoReport_v1_20260215_103000.json
│       └── EchoReport_v2_20260215_104500.json
├── optimized/              # Optimized DSPy programs (.json files)
│   ├── DocumentExtractor_optimized.json
│   └── RadiologyClassifier_optimized.json
├── checkpoints/            # Batch processing checkpoints
│   └── batch_20260217_090000/
│       ├── state.json      # Resume state
│       └── processed.txt   # List of processed files
└── logs/                   # Application logs
    └── mosaicx.log         # Debug and error logs
```

### Schema Storage

Schemas generated with `mosaicx schema generate` are saved as JSON files in `~/.mosaicx/schemas/`. Each schema is named after its class (e.g., `EchoReport.json`).

Whenever you refine a schema with `mosaicx schema refine`, the old version is automatically archived to `~/.mosaicx/schemas/.history/` with a timestamp. You can view version history with:

```bash
mosaicx schema history EchoReport
```

And revert to a previous version with:

```bash
mosaicx schema revert EchoReport --version 1
```

### Optimized Programs

When you optimize a DSPy pipeline with `mosaicx optimize`, the optimized program is saved as a JSON file in `~/.mosaicx/optimized/`. These files contain learned prompts and few-shot examples that improve extraction accuracy.

Load an optimized program with:

```bash
mosaicx extract --document report.pdf --optimized ~/.mosaicx/optimized/DocumentExtractor_optimized.json
```

### Checkpoints

Batch processing checkpoints are saved to `~/.mosaicx/checkpoints/` (or a custom directory specified with `--checkpoint-dir`). Checkpoints let you resume interrupted batch jobs without reprocessing files.

Resume a batch job with:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured --mode radiology --resume
```

### Logs

Application logs are written to `~/.mosaicx/logs/mosaicx.log`. Check this file if you encounter errors or need to debug issues.

## LLM Backends

MOSAICX can connect to any OpenAI-compatible LLM endpoint. This includes local inference servers (Ollama, llama.cpp, vLLM, SGLang, vLLM-MLX) and cloud APIs (OpenAI, Together AI, Groq, Anthropic).

### Ollama (Default — Local, Private)

Ollama is the default backend and requires no configuration. It runs large language models locally on your machine, ensuring that no patient data leaves your computer.

**Installation:**

- **macOS:** `brew install ollama` or download from [ollama.com](https://ollama.com)
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows:** Download from [ollama.com](https://ollama.com)

**Start the server:**

```bash
ollama serve
```

Leave this terminal window open while using MOSAICX.

**Pull a model:**

```bash
# For 16 GB RAM systems (fast, good quality)
ollama pull gpt-oss:20b

# For 64+ GB RAM systems (best quality)
ollama pull gpt-oss:120b
```

Verify the model is available:

```bash
ollama list
```

**Configuration:**

No configuration needed — MOSAICX is preconfigured to talk to Ollama on `localhost:11434`. Just make sure the server is running.

### llama.cpp (Local)

llama.cpp is a lightweight inference server that runs GGUF-format models locally. It is faster than Ollama for single-threaded inference and supports a wider range of quantization formats.

**Start the server:**

```bash
llama-server -m /path/to/model.gguf --port 8080
```

**Configuration:**

```bash
export MOSAICX_LM=openai/your-model
export MOSAICX_API_BASE=http://localhost:8080/v1
export MOSAICX_API_KEY=dummy
```

The model name can be anything (llama.cpp only loads one model at a time, so the name doesn't matter).

**Example:**

```bash
# Start llama.cpp server
llama-server -m gpt-oss-20b-Q4_K_M.gguf --port 8080

# Configure MOSAICX
export MOSAICX_LM=openai/gpt-oss-20b
export MOSAICX_API_BASE=http://localhost:8080/v1
export MOSAICX_API_KEY=dummy

# Run extraction
mosaicx extract --document report.pdf
```

### vLLM (GPU Server)

vLLM is a high-throughput GPU inference server optimized for large language models. It uses continuous batching and advanced memory management to handle many concurrent requests efficiently.

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

**On your local machine:**

```bash
# 1. SSH tunnel the server port to localhost
ssh -L 8000:localhost:8000 user@gpu-server

# 2. Configure MOSAICX
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy

# 3. Run extraction
mosaicx extract --document report.pdf
```

**Batch processing:**

Match `--workers` to the server's `--max-num-seqs` for optimal throughput:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured \
  --mode radiology --workers 4
```

### SGLang (GPU Server)

SGLang is another high-performance GPU inference server with similar capabilities to vLLM. It supports continuous batching and advanced scheduling.

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

**On your local machine:**

```bash
# 1. SSH tunnel the server port to localhost
ssh -L 30000:localhost:30000 user@gpu-server

# 2. Configure MOSAICX
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:30000/v1
export MOSAICX_API_KEY=dummy

# 3. Run extraction
mosaicx extract --document report.pdf
```

**Batch processing:**

Match `--workers` to the server's `--max-running-requests` for optimal throughput:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured \
  --mode radiology --workers 4
```

### vLLM-MLX (Apple Silicon)

vLLM-MLX is a vLLM-compatible inference server optimized for Apple Silicon Macs. It runs natively using MLX and unified memory, allowing you to run quantized models up to ~70B on a 128 GB Mac without needing a remote GPU server.

**Installation:**

```bash
uv tool install git+https://github.com/waybarrios/vllm-mlx.git
```

**Start the server:**

```bash
# For 16-32 GB RAM Macs (fast, good quality)
vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8 --port 8000

# For 64-128 GB RAM Macs (best quality)
vllm-mlx serve mlx-community/gpt-oss-120b-4bit --port 8000

# With continuous batching for concurrent requests (recommended)
vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8 --port 8000 --continuous-batching
```

Models are loaded from the `mlx-community` organization on Hugging Face.

**Configuration:**

```bash
export MOSAICX_LM=openai/mlx-community/gpt-oss-20b-MXFP4-Q8
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy
```

**Important:** Set `MOSAICX_LM` to `openai/<model-id>` matching the exact model you loaded (e.g., `openai/mlx-community/gpt-oss-20b-MXFP4-Q8`).

**Do NOT use `--reasoning-parser`** — MOSAICX strips Harmony channel tokens client-side, so server-side parsing is not needed.

**Example:**

```bash
# Start vLLM-MLX server
vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8 --port 8000 --continuous-batching

# Configure MOSAICX
export MOSAICX_LM=openai/mlx-community/gpt-oss-20b-MXFP4-Q8
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy

# Run extraction
mosaicx extract --document report.pdf
```

### Cloud APIs (OpenAI, Together AI, Groq)

MOSAICX can connect to cloud-based LLM APIs like OpenAI, Together AI, and Groq. Note that using cloud APIs sends your data to third-party servers, so ensure you comply with privacy regulations (HIPAA, GDPR) before using them with patient data.

**OpenAI:**

```bash
export MOSAICX_LM=openai/gpt-4o
export MOSAICX_API_KEY=sk-...
export MOSAICX_API_BASE=https://api.openai.com/v1
```

**Together AI:**

```bash
export MOSAICX_LM=together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
export MOSAICX_API_KEY=your-together-ai-key
export MOSAICX_API_BASE=https://api.together.xyz/v1
```

**Groq:**

```bash
export MOSAICX_LM=groq/llama-3.1-70b-versatile
export MOSAICX_API_KEY=gsk_...
export MOSAICX_API_BASE=https://api.groq.com/openai/v1
```

**Example:**

```bash
# Configure for OpenAI
export MOSAICX_LM=openai/gpt-4o
export MOSAICX_API_KEY=sk-proj-...
export MOSAICX_API_BASE=https://api.openai.com/v1

# Run extraction
mosaicx extract --document report.pdf
```

## OCR Configuration

MOSAICX includes two OCR engines that can be used separately or together:

### Surya (Traditional OCR)

Surya is a traditional OCR engine that uses layout detection and text recognition. It is fast and works well for clean printed text.

**When to use:**
- Clean, printed medical reports
- PDFs with poor or no text layer
- Fast OCR needed

**Configuration:**

```bash
export MOSAICX_OCR_ENGINE=surya
```

### Chandra (VLM-based OCR)

Chandra uses a Vision-Language Model (Qwen3-VL 9B) for OCR. It excels at handwriting, complex layouts, and tables.

**When to use:**
- Handwritten clinical notes
- Complex layouts with tables and annotations
- Poor-quality scans

**Configuration:**

```bash
export MOSAICX_OCR_ENGINE=chandra
export MOSAICX_CHANDRA_BACKEND=auto  # or "vllm" (GPU) or "hf" (CPU)
```

**Backends:**

- `vllm` — Fast, GPU-only (requires CUDA-compatible GPU)
- `hf` — CPU-compatible (slower, works on any machine)
- `auto` — Auto-detect (tries vLLM first, falls back to HF)

### Both Engines (Default)

By default, MOSAICX runs both Surya and Chandra in parallel, scores each output per page, and picks the best result.

**Configuration:**

```bash
export MOSAICX_OCR_ENGINE=both
```

This provides the best accuracy but uses more computational resources.

### OCR Quality Threshold

The `MOSAICX_QUALITY_THRESHOLD` setting controls the minimum acceptable OCR quality per page (0.0 to 1.0). Pages below this threshold trigger a warning.

```bash
export MOSAICX_QUALITY_THRESHOLD=0.6  # Default
```

Lower values accept lower-quality OCR results, while higher values enforce stricter quality requirements.

### Force OCR on Native PDFs

By default, MOSAICX extracts embedded text from native PDFs (PDFs with a text layer). If you want to force OCR even on native PDFs, set:

```bash
export MOSAICX_FORCE_OCR=true
```

This is useful when the embedded text is poorly formatted or unreliable.

### OCR Language Hints

The `MOSAICX_OCR_LANGS` setting provides language hints to the OCR engines, improving accuracy for non-English text.

```bash
export MOSAICX_OCR_LANGS=en,de,fr  # English, German, French
```

Use ISO 639-1 language codes (e.g., `en` for English, `de` for German, `fr` for French, `es` for Spanish).

## Supported Document Formats

MOSAICX supports the following document formats:

| Format | Extensions | OCR Support |
|--------|------------|-------------|
| **Text** | `.txt`, `.md` | N/A (plain text) |
| **PDF** | `.pdf` | Yes (native text extraction + OCR fallback) |
| **Word** | `.docx`, `.pptx` | No (text extraction only) |
| **Images** | `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp` | Yes (OCR applied automatically) |

### How MOSAICX Handles PDFs

1. Check if the PDF has embedded text (native PDF)
2. If yes, extract the text directly (fast, no OCR)
3. If no (scanned PDF), run OCR (Surya, Chandra, or both)
4. If `MOSAICX_FORCE_OCR=true`, always run OCR regardless of text layer

## Export Formats

MOSAICX supports multiple export formats for batch processing:

| Format | Extension | Description | Use Case |
|--------|-----------|-------------|----------|
| **JSON** | `.json` | Individual JSON files per document | Single document extraction |
| **JSONL** | `.jsonl` | JSON Lines (one JSON object per line) | Streaming, log analysis, LLM fine-tuning |
| **Parquet** | `.parquet` | Columnar format optimized for analytics | Data analysis with pandas, DuckDB, Spark |
| **CSV** | `.csv` | Comma-separated values with flattening | Excel, spreadsheets, legacy systems |

### Setting Default Export Formats

For batch processing, MOSAICX exports to JSONL and Parquet by default. You can change this with:

```bash
export MOSAICX_DEFAULT_EXPORT_FORMATS=jsonl,parquet
```

Or specify formats per-batch with the `--format` flag:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured \
  --mode radiology --format jsonl --format csv
```

### Export Format Details

**JSONL:**

```jsonl
{"patient_name": "John Doe", "exam_date": "2026-02-15", "_source": "report_001"}
{"patient_name": "Jane Smith", "exam_date": "2026-02-16", "_source": "report_002"}
```

One JSON object per line. Ideal for streaming and log-style processing.

**Parquet:**

Columnar format optimized for analytical queries. Requires `pandas` and `pyarrow`:

```bash
pip install pandas pyarrow
```

Load in Python:

```python
import pandas as pd
df = pd.read_parquet("results.parquet")
print(df.head())
```

**CSV:**

Flattened CSV with nested fields converted to dot-separated columns (e.g., `extracted.patient_name`).

## Common Configuration Scenarios

### Scenario 1: Local Ollama on macOS

**Goal:** Run MOSAICX entirely locally with no internet connection required.

**Configuration:**

No configuration needed — defaults work out-of-the-box. Just make sure Ollama is running:

```bash
ollama serve
ollama pull gpt-oss:20b
mosaicx extract --document report.pdf
```

### Scenario 2: Remote vLLM Server on DGX Spark

**Goal:** Connect MOSAICX to a remote GPU server running vLLM for faster inference.

**On the GPU server:**

```bash
vllm serve gpt-oss:120b --port 8000
```

**On your local machine:**

```bash
# SSH tunnel
ssh -L 8000:localhost:8000 user@dgx-spark

# Configure MOSAICX
export MOSAICX_LM=openai/gpt-oss:120b
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy

# Run batch processing with parallel workers
mosaicx batch --input-dir ./reports --output-dir ./structured \
  --mode radiology --workers 4
```

### Scenario 3: Apple Silicon Mac with vLLM-MLX

**Goal:** Run fast local inference on an Apple Silicon Mac using vLLM-MLX.

**Configuration:**

```bash
# Install vLLM-MLX
uv tool install git+https://github.com/waybarrios/vllm-mlx.git

# Start the server
vllm-mlx serve mlx-community/gpt-oss-20b-MXFP4-Q8 --port 8000 --continuous-batching

# Configure MOSAICX
export MOSAICX_LM=openai/mlx-community/gpt-oss-20b-MXFP4-Q8
export MOSAICX_API_BASE=http://localhost:8000/v1
export MOSAICX_API_KEY=dummy

# Run extraction
mosaicx extract --document report.pdf
```

### Scenario 4: OpenAI GPT-4o for Prototyping

**Goal:** Use OpenAI's GPT-4o for quick prototyping (not for production with PHI).

**Configuration:**

```bash
export MOSAICX_LM=openai/gpt-4o
export MOSAICX_API_KEY=sk-proj-...
export MOSAICX_API_BASE=https://api.openai.com/v1

mosaicx extract --document report.pdf
```

### Scenario 5: High-Quality OCR with Chandra Only

**Goal:** Use only Chandra for OCR (best for handwritten notes and complex layouts).

**Configuration:**

```bash
export MOSAICX_OCR_ENGINE=chandra
export MOSAICX_CHANDRA_BACKEND=auto
export MOSAICX_QUALITY_THRESHOLD=0.7

mosaicx extract --document handwritten_note.pdf
```

## Troubleshooting

### Error: "No API key configured"

**Cause:** `MOSAICX_API_KEY` is not set or is empty.

**Solution:**

```bash
export MOSAICX_API_KEY=ollama  # For Ollama
export MOSAICX_API_KEY=sk-...  # For OpenAI
```

### Error: Connection refused when connecting to LLM server

**Cause:** The LLM server is not running or is not reachable.

**Solution:**

1. Check that the server is running:
   - Ollama: `ollama serve`
   - vLLM: `vllm serve model-name --port 8000`
   - llama.cpp: `llama-server -m model.gguf --port 8080`

2. Check that the port matches `MOSAICX_API_BASE`:
   - Ollama: `:11434`
   - vLLM: `:8000`
   - llama.cpp: `:8080`
   - SGLang: `:30000`

3. If using SSH tunneling, make sure the tunnel is active:
   ```bash
   ssh -L 8000:localhost:8000 user@gpu-server
   ```

### Warning: Low OCR quality detected

**Cause:** The OCR confidence score is below `MOSAICX_QUALITY_THRESHOLD`.

**Solution:**

- Try using `MOSAICX_OCR_ENGINE=chandra` for better OCR quality
- Lower the threshold: `export MOSAICX_QUALITY_THRESHOLD=0.5`
- Improve the input document quality (scan at higher DPI, better lighting)

### Out of memory (OOM) during batch processing

**Cause:** Too many parallel workers for available RAM/VRAM.

**Solution:**

- Reduce `--workers` (e.g., from 8 to 4 or 2)
- On GPU servers, reduce `--max-num-seqs` (vLLM) or `--max-running-requests` (SGLang)
- Use a smaller model (e.g., `gpt-oss:20b` instead of `gpt-oss:120b`)

### Slow extraction performance

**Cause:** Using a large model on limited hardware, or using a single worker.

**Solution:**

- Use a smaller/faster model (e.g., `gpt-oss:20b`)
- For batch processing, increase `--workers` (if you have sufficient RAM)
- Use a dedicated GPU server (vLLM, SGLang) for faster inference
- Use vLLM-MLX on Apple Silicon for faster local inference

## Advanced Topics

### Combining CLI Flags and Environment Variables

You can mix CLI flags and environment variables. CLI flags take precedence:

```bash
# Set default in .env
export MOSAICX_BATCH_WORKERS=8

# Override with CLI flag
mosaicx batch --input-dir ./reports --output-dir ./structured --workers 8
```

### Tuning Batch Workers for Optimal Throughput

The optimal number of workers depends on your hardware and LLM backend:

| Backend | Recommended Workers | Notes |
|---------|---------------------|-------|
| Ollama | 1-2 | Ollama handles concurrency internally |
| llama.cpp | 1 | Single-threaded by design |
| vLLM | Match `--max-num-seqs` | Set workers = max concurrent requests on server |
| SGLang | Match `--max-running-requests` | Set workers = max concurrent requests on server |
| vLLM-MLX | 2-4 | Depends on Mac RAM and model size |
| OpenAI / Cloud APIs | 4-8 | Rate-limited by API provider |

**Rule of thumb:** Start with 2-4 workers and increase gradually while monitoring memory usage. If you encounter OOM errors, reduce workers.

### Optimizing OCR for Specific Languages

For non-English documents, provide language hints to improve OCR accuracy:

```bash
# German + English
export MOSAICX_OCR_LANGS=de,en

# French + English
export MOSAICX_OCR_LANGS=fr,en

# Spanish + Portuguese + English
export MOSAICX_OCR_LANGS=es,pt,en
```

Use ISO 639-1 language codes. The first language in the list is given higher priority.

## Summary

MOSAICX is highly configurable, but the defaults work out-of-the-box for local Ollama usage. For advanced setups (remote GPU servers, cloud APIs, custom OCR settings), use environment variables or a `.env` file to override defaults.

Key takeaways:

1. **No configuration needed** if using Ollama on localhost
2. **Use `.env` files** for persistent configuration
3. **View current config** with `mosaicx config show`
4. **Override defaults** with `MOSAICX_*` environment variables
5. **Match workers to server concurrency** for optimal batch processing
6. **Use OCR language hints** for non-English documents
7. **Choose the right backend** based on privacy, performance, and hardware

For further help, see the [Getting Started guide](getting-started.md) or report issues at [github.com/LalithShiyam/MOSAICX/issues](https://github.com/LalithShiyam/MOSAICX/issues).

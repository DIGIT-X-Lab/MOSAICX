# Getting Started with MOSAICX

Welcome to MOSAICX — a tool that turns unstructured medical documents (like radiology reports, clinical notes, and pathology summaries) into validated, machine-readable JSON data.

This guide assumes you are a complete beginner who has never used a command-line tool, Python, or Ollama before. We will walk through everything step by step.

## Prerequisites

Before you start, you will need:

- **Python 3.11 or higher** — a programming language that MOSAICX is built with
- **A terminal** — a text-based interface to run commands (macOS: Terminal.app, Linux: Terminal, Windows: PowerShell or Windows Terminal)
- **At least 16 GB RAM** — minimum for running local LLM models (64 GB recommended for best performance)
- **Internet connection** — for downloading software and models

### What is Python?

Python is a programming language used to build software. MOSAICX is written in Python, so you need Python installed on your computer to run it.

To check if Python is already installed, open your terminal and run:

```bash
python3 --version
```

You should see output like `Python 3.11.5` or `Python 3.12.2`. If you see an error or a version older than 3.11, you need to install or update Python:

- **macOS:** Install using [Homebrew](https://brew.sh/): `brew install python@3.11`
- **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install python3.11`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

### What is Ollama?

Ollama is a local server that runs large language models (LLMs) on your machine. Think of it as a local version of ChatGPT that processes your medical reports entirely on your computer — no data leaves your machine, ensuring patient privacy.

MOSAICX uses Ollama by default because it is easy to set up and keeps your data private.

## Step 1: Install Ollama

Ollama is the local LLM server that MOSAICX talks to by default.

### macOS

Option 1 — using Homebrew (if you have Homebrew installed):

```bash
brew install ollama
```

Option 2 — download from the website:

1. Go to [ollama.com](https://ollama.com)
2. Click "Download for macOS"
3. Open the downloaded file and follow the installation instructions

### Linux

Run the official install script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This will download and install Ollama on your system.

### Windows

1. Go to [ollama.com](https://ollama.com)
2. Click "Download for Windows"
3. Run the installer and follow the on-screen instructions

### Start the Ollama server

After installing Ollama, you need to start the server. Open a new terminal window and run:

```bash
ollama serve
```

You should see output like:

```
Ollama is running
Listening on 127.0.0.1:11434
```

> **Important:** Leave this terminal window open while using MOSAICX. The server needs to keep running in the background.

### Pull a model

Ollama needs to download a language model before you can use it. MOSAICX recommends using the `gpt-oss` models, which are optimized for medical text extraction.

Open a **new** terminal window (keep the `ollama serve` terminal running) and run one of these commands:

**For 16 GB RAM systems (fast, good quality):**

```bash
ollama pull gpt-oss:20b
```

This downloads a 20-billion parameter model (about 12 GB). It is fast and works well for most medical documents.

**For 64+ GB RAM systems (best quality):**

```bash
ollama pull gpt-oss:120b
```

This downloads a 120-billion parameter model (about 64 GB). It provides the best extraction accuracy but requires more memory.

The download may take 10-30 minutes depending on your internet speed. Once complete, you can verify the model is available:

```bash
ollama list
```

You should see `gpt-oss:20b` or `gpt-oss:120b` in the list.

## Step 2: Install MOSAICX

Now that Ollama is set up, install MOSAICX itself.

### Using pip (recommended for beginners)

Open a terminal and run:

```bash
pip install mosaicx
```

> **Note:** On some systems you may need to use `pip3` instead of `pip`:
>
> ```bash
> pip3 install mosaicx
> ```

### Alternative installation methods

If you prefer other package managers:

**Using uv (fast Python package manager):**

```bash
uv add mosaicx
```

**Using pipx (installs CLI tools in isolation):**

```bash
pipx install mosaicx
```

### Verify installation

Check that MOSAICX is installed correctly:

```bash
mosaicx --version
```

You should see output like:

```
MOSAICX 2.0.0a1
```

> **No .env file needed:** MOSAICX is preconfigured to talk to Ollama on `localhost:11434`. If you followed Step 1, everything should just work. You only need to configure environment variables if you are using a different LLM backend (covered in the advanced configuration docs).

## Step 3: Your First Extraction

Let's extract structured data from a medical document. You can use a text file or PDF — MOSAICX supports both.

### Auto mode (recommended for beginners)

Auto mode is the simplest way to extract data. MOSAICX reads your document and automatically figures out what information to extract based on the content.

**Example:** Extract from a text file:

```bash
mosaicx extract --document report.txt
```

**Example:** Extract from a PDF:

```bash
mosaicx extract --document ct_scan_report.pdf
```

### What the output looks like

After a few seconds (or minutes, depending on document length and model size), you will see output in your terminal like this:

```json
{
  "extracted": {
    "patient_name": "John Doe",
    "exam_date": "2026-02-15",
    "exam_type": "CT Chest",
    "findings": "No acute findings. Lungs are clear.",
    "impression": "Normal chest CT."
  }
}
```

This is your medical document transformed into structured JSON data.

### Saving output to a file

To save the extracted data instead of printing it to the terminal, use the `-o` or `--output` flag:

```bash
mosaicx extract --document report.pdf -o output.json
```

Now the structured data is saved to `output.json` in your current directory. You can open this file in any text editor or use it for further processing.

## Step 4: Using Specialized Modes

MOSAICX includes built-in "modes" — specialized multi-step extraction pipelines optimized for specific document types.

### Radiology mode

The radiology mode is a 5-step pipeline designed specifically for radiology reports:

1. **Classify the exam type** (e.g., CT Chest, MRI Brain)
2. **Parse sections** (Technique, Findings, Impression)
3. **Extract technique details** (contrast, protocol)
4. **Extract findings** (measurements, anatomy codes)
5. **Extract impression** (summary, recommendations)

To use radiology mode:

```bash
mosaicx extract --document ct_report.pdf --mode radiology
```

The output will include additional fields specific to radiology reports, such as anatomy codes, measurement scores (BI-RADS, Lung-RADS), and structured findings.

### Pathology mode

The pathology mode is a 5-step pipeline for pathology reports:

1. **Classify the specimen type**
2. **Parse sections**
3. **Extract histology** (tumor type, grade)
4. **Extract TNM staging**
5. **Extract biomarkers** (e.g., HER2, ER, PR status)

To use pathology mode:

```bash
mosaicx extract --document pathology_report.pdf --mode pathology
```

### List available modes

To see all available modes:

```bash
mosaicx extract --list-modes
```

This prints a table showing each mode name and its description.

## Step 5: Schema-Based Extraction

Schemas let you define exactly what data you want to extract. Think of a schema as a template that tells MOSAICX what fields to look for and what data types to use.

### Generate a schema

Describe what you want to extract in plain English, and MOSAICX will create a schema for you:

```bash
mosaicx schema generate --description "echocardiography report with LVEF, valve grades, and clinical impression"
```

MOSAICX will generate a schema (a Pydantic model) and save it to `~/.mosaicx/schemas/`. You should see output like:

```
✓ Schema generated — it's alive!
Model: EchoReport
Fields: lvef, valve_grades, clinical_impression
Saved: /Users/yourusername/.mosaicx/schemas/EchoReport.json
```

You can give the schema a custom name:

```bash
mosaicx schema generate \
  --description "patient vitals including blood pressure, heart rate, temperature" \
  --name PatientVitals
```

### List your schemas

See all saved schemas:

```bash
mosaicx schema list
```

This prints a table of schema names, the number of fields, and descriptions.

### Extract with a schema

Once you have generated a schema, use it to extract data from a document:

```bash
mosaicx extract --document echo_report.pdf --schema EchoReport
```

MOSAICX will extract only the fields defined in the `EchoReport` schema, ensuring consistent output structure across multiple documents.

### Refine a schema

After using a schema, you may realize you need to add or remove fields. Use the `schema refine` command:

**Add a required field:**

```bash
mosaicx schema refine --schema EchoReport --add "rvsp: float"
```

**Add an optional field with a description:**

```bash
mosaicx schema refine --schema EchoReport \
  --add "hospital_name: str" \
  --optional \
  --description "Name of the treating hospital"
```

**Remove a field:**

```bash
mosaicx schema refine --schema EchoReport --remove clinical_impression
```

**Rename a field:**

```bash
mosaicx schema refine --schema EchoReport --rename "lvef=lvef_percent"
```

**Let the LLM refine the schema:**

```bash
mosaicx schema refine --schema EchoReport \
  --instruction "add a field for pericardial effusion severity as an enum with values mild, moderate, severe"
```

Every time you refine a schema, MOSAICX automatically saves the old version to `.history/`, so you never lose your work.

## Step 6: Batch Processing

If you have many documents to process, use batch mode instead of running `extract` on each file individually.

### Basic batch processing

Suppose you have a folder full of radiology reports at `./reports/`. To process all of them:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured
```

MOSAICX will:

1. Find all supported documents in `./reports/` (`.txt`, `.pdf`, `.docx`, images)
2. Extract data from each one
3. Save the results as JSON files in `./structured/`

### Batch with a mode

To use a specialized mode (e.g., radiology) for all documents:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured --mode radiology
```

### Batch with a schema

To extract using a specific schema:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured --schema EchoReport
```

### Resume after interruption

If batch processing stops (e.g., your computer went to sleep), you can resume where you left off:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured --mode radiology --resume
```

MOSAICX will skip files that have already been processed.

### Parallel processing

By default, MOSAICX processes one document at a time. You can process multiple documents in parallel to speed things up:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured --mode radiology --workers 4
```

The `--workers` flag tells MOSAICX how many documents to process at once. Start with 2-4 workers and increase if your system has enough RAM and CPU cores.

> **Important:** Parallel processing uses more memory. If you run out of memory, reduce the number of workers.

### Export formats

By default, batch mode creates individual JSON files for each document. You can also export as JSONL (JSON Lines) or Parquet:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured \
  --mode radiology --format jsonl --format parquet
```

- **JSONL:** One JSON object per line, useful for streaming and log analysis
- **Parquet:** Columnar format optimized for analytics (requires pandas + pyarrow)

## What's Next?

You have learned the basics of MOSAICX. Here are some advanced topics to explore:

- **[Schemas and Templates](schemas-and-templates.md)** — Learn how to create custom schemas and use YAML templates for extraction
- **[Pipelines](pipelines.md)** — Deep dive into the radiology and pathology pipelines, and how to create your own
- **[Optimization](optimization.md)** — Tune DSPy pipelines with labeled examples for better accuracy
- **[Configuration](configuration.md)** — Customize MOSAICX settings, connect to different LLM backends, configure OCR engines

## Common Questions

### How do I stop the Ollama server?

In the terminal window where you ran `ollama serve`, press `Ctrl+C` (or `Cmd+C` on macOS).

### What if I get an error "No API key configured"?

MOSAICX is preconfigured to use Ollama with the API key `ollama`. If you are seeing this error, it means the configuration is not being read correctly. Make sure:

1. Ollama is running (`ollama serve` in a separate terminal)
2. You are not setting `MOSAICX_API_KEY` to an empty value

### What file formats are supported?

MOSAICX supports:

- **Text:** `.txt`, `.md`
- **PDF:** `.pdf` (with OCR if scanned)
- **Word:** `.docx`
- **Images:** `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` (OCR applied automatically)

### Where are schemas saved?

Schemas are saved to `~/.mosaicx/schemas/` as JSON files. Each schema is named after its class (e.g., `EchoReport.json`).

### Can I use MOSAICX without Ollama?

Yes. MOSAICX can connect to any OpenAI-compatible LLM endpoint, including:

- **OpenAI** (gpt-4, gpt-4o)
- **llama.cpp** (local or remote server)
- **vLLM** (GPU inference server)
- **SGLang** (fast inference server)
- **vLLM-MLX** (Apple Silicon optimized)

See the [Configuration guide](configuration.md) for details on switching backends.

### How do I update MOSAICX?

To update to the latest version:

```bash
pip install --upgrade mosaicx
```

Or if you used `pipx`:

```bash
pipx upgrade mosaicx
```

### Where can I get help?

- **Documentation:** Browse the `/docs` folder in the MOSAICX repository
- **Issues:** Report bugs or request features at [github.com/LalithShiyam/MOSAICX/issues](https://github.com/LalithShiyam/MOSAICX/issues)
- **Email:** Research inquiries to [lalith.shiyam@med.uni-muenchen.de](mailto:lalith.shiyam@med.uni-muenchen.de)

---

Ready to structure your medical documents? Start with Step 1 and work your way through. If you get stuck, refer back to this guide or reach out for help.

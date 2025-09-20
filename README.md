<div align="center">
  <img src="assets/mosaicx_logo.png" alt="MOSAICX Logo" width="800"/>
</div>
<p align="center">
  <a href="https://pypi.org/project/mosaicx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mosaicx.svg?label=PyPI&style=flat-square&logo=python&logoColor=white&color=bd93f9"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-50fa7b?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img alt="License" src="https://img.shields.io/badge/License-AGPL--3.0-ff79c6?style=flat-square&logo=gnu&logoColor=white"></a>
  <a href="https://pepy.tech/project/mosaicx"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mosaicx?style=flat-square&color=8be9fd&label=Downloads"></a>
  <a href="https://pydantic.dev"><img alt="Pydantic v2" src="https://img.shields.io/badge/Pydantic-v2-ffb86c?style=flat-square&logo=pydantic&logoColor=white"></a>
  <a href="https://ollama.ai"><img alt="Ollama Compatible" src="https://img.shields.io/badge/Ollama-Compatible-6272a4?style=flat-square&logo=ghost&logoColor=white"></a>
</p>

# MOSAICX: Structure first. Then everything else.

MOSAICX turns unstructured clinical documents into **validated, structured data**—locally, privately, reproducibly. It supports:

- **Schema generation** from natural language (Pydantic v2)  
- **Extraction** from PDFs/text using the generated schema  
- **Summarization** of radiology reports (single or multi-report per patient) → **critical timeline + one-paragraph executive summary** as **JSON**

> Local LLMs via **Ollama** (OpenAI-compatible). PDF text via **Docling**. Rich terminal UI.

---

## 🚀 Quick Start

### 1) Requirements
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model that behaves well with JSON
ollama pull llama3.1:8b-instruct         # or: qwen2.5:7b-instruct, gpt-oss:120b
```

### 2) Install MOSAICX
```bash
pip install mosaicx
# or (faster resolver)
uv add mosaicx
```

### 3) Smoke test
```bash
mosaicx --help
```

---

## ✨ New: Summarize (Timeline + JSON)

**Goal:** give clinicians an at-a-glance patient trajectory from one or more radiology reports (same patient), without reading everything.

- **Input:** one or many reports (`.pdf` or `.txt`) for a single patient  
- **Logic:** radiology-first prompt (modality-adaptive), concise, **no recommendations/differentials**  
- **Output:**  
  - **Terminal**: header + timeline + executive summary (Rich)  
  - **JSON**: standardized object (`patient`, `timeline[]`, `overall`)

**Example**
```bash
mosaicx summarize \
  --report P001_CT_2025-08-01.pdf \
  --report P001_CT_2025-09-10.pdf \
  --patient P001 \
  --model llama3.1:8b-instruct \
  --json-out out/summary_P001.json \
```
**or**
```bash
mosaicx summarize \
  --dir ./patient_directory
  --json-out ./longitudinal_summary.json
  --model gpt-oss:120b
```

**Summary JSON (shape)**
```json
{
  "patient": {
    "patient_id": "P001",
    "dob": null,
    "sex": null,
    "last_updated": "2025-09-19T12:34:56Z"
  },
  "timeline": [
    { "date": "2025-08-01", "source": "CT 2025-08-01", "note": "Baseline nodal disease; R ext-iliac LN short-axis 12 mm" },
    { "date": "2025-09-10", "source": "CT 2025-09-10", "note": "R ext-iliac LN 12→16 mm — progression; no visceral mets" }
  ],
  "overall": "Nodal-only disease with interval progression of the right external iliac node [CT 2025-09-10]; baseline nodal disease without visceral metastases [CT 2025-08-01]."
}
```

**Under the hood (robust fallbacks)**  
1) Instructor JSON → Pydantic → ✅  
2) Raw JSON extraction → Pydantic → ✅  
3) Heuristic timeline/summary → ✅  

---

## Core Workflows

### 1) Generate a schema (from plain English)
```bash
mosaicx generate \
  --desc "Echocardiography with patient_id, exam_date, EF %, valve grades (Normal/Mild/Moderate/Severe), impression" \
  --model gpt-oss:120b
```

### 2) Extract structured data with that schema(PDF → JSON)
```bash
mosaicx extract \
  --pdf echo_report.pdf \
  --schema EchocardiographyReport_20250919_143022 \
  --model gpt-oss:120b \
  --save out/echo_001.json
```

### 3) Summarize radiology reports (timeline + JSON)
```bash
# Multiple inputs for the same patient
mosaicx summarize \
  --dir ./reports/P001 \
  --patient P001 \
  --model gpt-oss:120b \
  --json-out out/summary_P001.json \
```

**CLI options (summarize)**  
- `--report` … (repeatable)  
- `--dir` (recursively picks `.pdf`, `.txt`)  
- `--patient PSEUDONYM`  
- `--json-out path.json` and `--print-json`  
- `--model`, `--base-url`, `--api-key`, `--temperature`

---

## Tips for Great Results

- **Models:** prefer `llama3.1:8b-instruct` or `qwen2.5:7b-instruct` for clean JSON.  
- **Prompts (summarize):** MOSAICX uses a **conciseness-first** prompt, modality-adaptive, no DDx or recommendations.  
- **PDFs:** If scanned (no text), run OCR before MOSAICX or add an OCR pre-step in your pipeline.

---

## Troubleshooting

- **Connection refused / model not found**: start Ollama, `ollama list`, pull your model.  
- **Empty summary**: try a more JSON-obedient model; lower `--temperature` (0.0–0.2).  
- **PDF yields no text**: the PDF likely has no text layer; OCR it first.

---

## Why MOSAICX (one paragraph)

MOSAICX is **infrastructure** for clinical data: schema-driven, validated, local, and reproducible. Structure reports once, then reuse the same schemas and summarizers across departments and time—enabling longitudinal analysis, cross-modal integration, and downstream intelligence without sending data to the cloud.

---

## License

AGPL-3.0. See `LICENSE`.

### Contact
DIGIT-X Lab · Department of Radiology · LMU Klinikum  
`lalith.shiyam@med.uni-muenchen.de`

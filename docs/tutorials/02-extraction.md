# Tutorial 2: Extracting Structured Data

Extract structured data from clinical documents using a template.

## Prerequisites

- A running LLM server (local or remote)
- A template (see [Tutorial 1](01-template-creation.md))

Configure your LLM in `.env`:

```bash
# Local server
MOSAICX_LM=mlx-community/gpt-oss-120b-4bit
MOSAICX_API_BASE=http://localhost:8000/v1

# Or remote server
MOSAICX_LM=openai/gpt-oss:120b
MOSAICX_API_BASE=https://your-server.com/ollama/v1
MOSAICX_API_KEY=your-key
```

## Basic Extraction

```bash
mosaicx extract --document report.pdf --template BasicExtraction -o result.json
```

The output JSON contains the extracted fields:

```json
{
  "extracted": {
    "patient_name": "Sarah Johnson",
    "patient_id": "PID-12345",
    "sex": "Female",
    "date_of_birth": "1985-03-15",
    "bmi": 23.4
  },
  "_evidence": {
    "patient_name": {
      "excerpt": "Patient Name: Sarah Johnson",
      "reasoning": "The document lists the patient name as Sarah Johnson."
    }
  }
}
```

Every field has an excerpt (where it was found) and reasoning (why the LLM chose this value).

## Auto Extraction (No Template)

Don't have a template? MOSAICX can infer the structure:

```bash
mosaicx extract --document report.pdf -o result.json
```

The LLM decides what fields to extract based on the document content.

## Extraction Modes

Use `--mode` for domain-specific multi-step extraction:

```bash
# Radiology reports (5-step pipeline)
mosaicx extract --document ct_scan.pdf --mode radiology -o result.json

# Pathology reports
mosaicx extract --document path_report.pdf --mode pathology -o result.json
```

## Think Levels

Control the extraction depth:

```bash
# Fast (default) -- structured output, no reasoning chain
mosaicx extract --document report.pdf --template MyTemplate --think fast -o result.json

# Auto -- picks the best strategy for the template complexity
mosaicx extract --document report.pdf --template MyTemplate --think auto -o result.json

# Deep -- chunked extraction + verification + fix pass
mosaicx extract --document report.pdf --template MyTemplate --think deep -o result.json
```

Fast is ~5s, deep is ~25s. Both produce excerpts and reasoning.

## Scanned Documents and Images

MOSAICX handles PDFs, scanned PDFs, and images (JPG, PNG, TIFF):

```bash
# Native text-layer PDF (fast, no OCR)
mosaicx extract --document text_report.pdf --template MyTemplate -o result.json

# Scanned PDF or image (automatic OCR via PPStructure)
mosaicx extract --document scan.jpg --template MyTemplate -o result.json

# Force OCR on a text-layer PDF (enables table detection)
mosaicx extract --document report.pdf --template MyTemplate --force-ocr -o result.json
```

Use `--force-ocr` when the PDF has tables -- PPStructure detects table structure and produces clean Markdown tables for the LLM.

## Debugging OCR Output

See exactly what the LLM receives:

```bash
mosaicx extract --document scan.pdf --template MyTemplate --dump-ocr -o result.json
```

Check the `.ocr.txt` file:

```bash
cat result.ocr.txt
```

If the OCR text is wrong, it's an OCR problem. If the OCR text is correct but the extraction is wrong, it's an LLM problem.

## Batch Processing

Extract from a directory of documents:

```bash
mosaicx extract --dir /path/to/documents/ --template MyTemplate --output-dir /path/to/output/
```

Results are saved as individual JSON files in the output directory, with optional export to JSONL, CSV, or Parquet.

## What's Next

To remove patient identifiers from documents, see [Tutorial 3: De-identification](03-deidentification.md).

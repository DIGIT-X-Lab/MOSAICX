# Tutorial 3: De-identifying Documents

Remove Protected Health Information (PHI) from clinical documents.

## Prerequisites

- A running LLM server (see [Tutorial 2](02-extraction.md) for setup)

## Basic De-identification

```bash
mosaicx deidentify --document patient_note.pdf -o redacted.pdf
```

Produces a redacted PDF with PHI replaced by black boxes.

## Output Formats

The output format depends on the file extension:

```bash
# Redacted PDF (PHI visually blacked out)
mosaicx deidentify --document report.pdf -o redacted.pdf

# Redacted image
mosaicx deidentify --document scan.jpg -o redacted.jpg

# JSON with redacted text + evidence
mosaicx deidentify --document report.pdf -o result.json
```

JSON output:

```json
{
  "redacted_text": "Patient Name: [REDACTED]\nPatient ID: [REDACTED]\n...",
  "mode": "remove",
  "_evidence": {
    "name_0": {
      "excerpt": "Patient Name: Sarah Johnson",
      "reasoning": "Patient name is protected health information"
    },
    "date_2": {
      "excerpt": "Date of Birth: March 15, 1985",
      "reasoning": "Date of birth is protected health information"
    }
  }
}
```

## De-identification Modes

```bash
# Remove (default) -- replace PHI with [REDACTED]
mosaicx deidentify --document report.pdf --mode remove -o redacted.pdf

# Pseudonymize -- replace with realistic fake values
mosaicx deidentify --document report.pdf --mode pseudonymize -o redacted.pdf

# Date shift -- shift all dates by a random offset
mosaicx deidentify --document report.pdf --mode dateshift -o redacted.pdf
```

## Scanned Documents

Works on scanned PDFs and images with automatic OCR:

```bash
# Scanned PDF (OCR automatic)
mosaicx deidentify --document scanned_report.pdf -o redacted.pdf

# Image
mosaicx deidentify --document lab_results.jpg -o redacted.jpg

# Force OCR for table-aware processing
mosaicx deidentify --document report.pdf --force-ocr -o redacted.pdf
```

## Debugging

See what the LLM receives and what PHI was detected:

```bash
mosaicx deidentify --document report.pdf --dump-ocr -o redacted.pdf
```

The CLI output shows:
- `01 -- REDACTED DOCUMENT`: the redacted text with `[REDACTED]` highlighted in coral
- `02 -- PHI DETECTED`: table of all PHI items with type, original text, and position

## Batch De-identification

```bash
mosaicx deidentify --dir /path/to/documents/ --output-dir /path/to/output/ --workers 4
```

## Provenance (Source Coordinates)

To get exact page + bounding box coordinates for each PHI item:

```bash
mosaicx deidentify --document report.pdf --provenance -o result.json
```

The `redaction_map` entries include `spans` with page numbers and bounding boxes for PDF overlay.

# Signals Integration (Internal)

Proprietary integration with the deepcOS Signals API for evaluating
AI-generated radiology reports against reference reports.

## Setup

Set the Signals API key as an environment variable:

```bash
export MOSAICX_SIGNALS_API_KEY="sk_signals_YOUR_KEY"
```

Optionally override the API base URL (defaults to the production endpoint):

```bash
export MOSAICX_SIGNALS_API_BASE="https://custom-endpoint/signals-api"
```

The `signals` command group is hidden from `mosaicx --help` and only
functions when `MOSAICX_SIGNALS_API_KEY` is set.

## CLI Usage

### Health Check

```bash
mosaicx signals health
```

### Evaluate a Report

Both reports are de-identified on-prem before being sent to Signals:

```bash
# Text files
mosaicx signals evaluate \
  --ai-report ai_report.txt \
  --reference ground_truth.txt

# PDFs
mosaicx signals evaluate \
  --ai-report ai_scan.pdf \
  --reference reference.pdf \
  --force-ocr

# With metadata
mosaicx signals evaluate \
  --ai-report ai_report.txt \
  --reference ground_truth.txt \
  --modality CT \
  --body-part chest \
  --model-name vlm-rad \
  --model-version 1.2

# Save full JSON response
mosaicx signals evaluate \
  --ai-report ai_report.txt \
  --reference ground_truth.txt \
  -o evaluation_result.json

# Skip de-identification (reports already clean)
mosaicx signals evaluate \
  --ai-report clean_ai.txt \
  --reference clean_ref.txt \
  --skip-deidentify
```

### Supported Input Formats

- Text: `.txt`, `.md`
- PDF: `.pdf` (native text extraction or OCR)
- Images: `.jpg`, `.png`, `.tiff`, `.bmp` (via OCR)

## SDK Usage

```python
from mosaicx.sdk import signals_evaluate

# From file paths
result = signals_evaluate(
    ai_report="ai_report.pdf",
    reference="ground_truth.txt",
    metadata={"modality": "CT", "body_part": "chest"},
)
print(result["trust_score"], result["verdict"])

# From raw text
result = signals_evaluate(
    ai_report_text="No acute cardiopulmonary abnormality.",
    reference_text="Small right pleural effusion noted.",
    metadata={"modality": "CR", "body_part": "chest"},
)

# Skip de-identification
result = signals_evaluate(
    ai_report_text="Already clean AI text.",
    reference_text="Already clean reference.",
    skip_deidentify=True,
)
```

## API Client (Direct)

For advanced usage, use the client directly:

```python
from mosaicx.signals import SignalsClient

client = SignalsClient(api_key="sk_signals_...")

# Health check
status = client.health()

# Evaluate
result = client.evaluate(
    generated_report="AI report text",
    reference_report="Reference text",
    metadata={"modality": "CT"},
)

# Retrieve past evaluation
past = client.status("evaluation-uuid")
```

## Response Schema

The evaluate endpoint returns:

| Field | Type | Description |
|-------|------|-------------|
| `evaluation_id` | string | UUID for this evaluation |
| `trust_score` | int | 0-100 trust score |
| `verdict` | string | `critical`, `review`, `acceptable`, `excellent` |
| `verdict_title` | string | Human-readable verdict label |
| `verdict_explanation` | string | Why this verdict was assigned |
| `trust_score_composition` | object | Component weights |
| `metrics` | object | BERTScore, RadGraph F1, CheXbert |
| `safety_signals` | array | Critical findings (missed, hallucinated) |
| `discrepancies` | array | Statement-level differences |
| `structured_review` | object | Scores (1-5), priority, risk, issues |
| `metadata` | object | Echo of input + PII detection flag |

## Error Handling

| Exception | HTTP | When |
|-----------|------|------|
| `SignalsAuthError` | 401 | Invalid or missing API key |
| `SignalsValidationError` | 400 | Bad request parameters |
| `SignalsRateLimitError` | 429 | Rate limit exceeded |
| `SignalsUpstreamError` | 502/504 | Transient AI service failure |
| `SignalsError` | other | General API error |

All exceptions have `.code`, `.status`, and `.message` attributes.
`SignalsRateLimitError` additionally has `.retry_after` (seconds).

## Rate Limits

| Tier | Daily | Burst | Access |
|------|-------|-------|--------|
| Free | 20/day | 2/min | On request |
| Developer | 500/day | 10/min | On request |
| Enterprise | Custom | 60/min | Contact sales |

Rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`,
`X-RateLimit-Reset`.

## Data Flow

```
AI Report ────┐
              ├── MOSAICX Deidentifier (on-prem) ──► Signals API ──► Trust Score
Reference ───┘       PHI never leaves machine          /evaluate       Verdict
                                                                       Safety Signals
```

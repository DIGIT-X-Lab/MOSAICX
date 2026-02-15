# Robust Document Parser Design

## Problem

The current document loader uses Docling for PDF/DOCX/PPTX and direct read for text files. Docling works well for native (digital) PDFs but struggles with scanned documents, handwritten notes, faxed referrals, and mixed-layout medical documents. For a medical document structuring platform, this is the "garbage in, garbage out" bottleneck — if the document text is garbled, the entire downstream pipeline produces garbage.

## Goal

Replace Docling with a dual-engine OCR system using **Surya** (fast, layout-aware) and **Chandra** (VLM-based, handles handwriting/forms/complex layouts). Run both in parallel on every document, quality-score per page, pick the best result. Accuracy-first design for medical documents.

## Architecture

```
Document
  |
  +-- .txt/.md --> direct read (instant, no OCR)
  |
  +-- .pdf/.docx/.pptx/.jpg/.png/.tiff
       |
       +---- Surya ---------------+
       |   (layout + OCR)         |
       |                          +-> Quality Scorer (per page) -> LoadedDocument
       +---- Chandra -------------+
            (VLM OCR)
```

Both engines run in parallel via ThreadPoolExecutor. The quality scorer compares results per page and picks the winner for each page independently — a 50-page document might have 48 typed pages (Surya wins) and 2 handwritten pages (Chandra wins).

## OCR Engines

### Surya

- **Role**: Fast-path OCR + layout detection
- **Architecture**: Traditional pipeline (detection -> recognition), ~200MB models
- **Strengths**: Fast (<1s/page), 90+ languages, good layout/table detection, CPU-capable
- **Weaknesses**: Struggles with handwriting, complex forms, low-quality scans
- **Platform**: PyTorch — works on CUDA (NVIDIA) and MPS (Mac) natively

### Chandra

- **Role**: Robust OCR for difficult documents
- **Architecture**: Fine-tuned Qwen3-VL 9B vision-language model, ~18GB
- **Strengths**: Handwriting (90.8% benchmark), forms/checkboxes, complex layouts, tables with merged cells, 40+ languages
- **Weaknesses**: Slower (~5-15s/page), requires GPU
- **Platform**: Auto-detect backend — vLLM on CUDA, HuggingFace transformers on Mac (MPS)
- **Output**: Markdown, HTML with bounding boxes, JSON with layout metadata

### Why both?

Surya is fast and accurate for typed/clean documents. Chandra handles everything but is slower. Running both and picking the best per page gives optimal accuracy across the full spectrum of medical documents (typed reports, scanned forms, handwritten notes, faxed referrals).

## LLM Inference (Separate from OCR)

The main DSPy pipelines (extraction, summarization, deidentification) use gpt-oss 20B and 120B models served via **llama.cpp** (`llama-server`):

- `llama-server` provides an OpenAI-compatible API
- DSPy/LiteLLM connects natively via the `openai/` prefix
- llama.cpp works on both Mac (Metal) and NVIDIA (CUDA)
- GGUF quantization is valuable for 120B model (~240GB fp16 -> ~60GB Q4_K_M)
- No code changes needed in MOSAICX — just configure `OPENAI_API_BASE`

This is completely separate from the OCR stack. Chandra uses its own inference (vLLM/HF), the LLM pipelines use llama.cpp.

## Quality Scorer

Deterministic scorer (no LLM needed) that evaluates OCR output quality:

| Metric | Weight | Description |
|--------|--------|-------------|
| Medical vocabulary hit rate | 0.35 | Count recognized medical terms against a curated ~500-term wordlist (anatomy, findings, modalities). Garbled OCR produces gibberish that won't match. |
| Character sanity | 0.25 | Ratio of alphanumeric + standard punctuation vs. total characters. Garbled OCR has unusual character distributions. |
| Word structure | 0.20 | Average word length between 2-15 chars, reasonable whitespace ratio. OCR failures produce run-together text or single-char fragments. |
| Text length | 0.20 | Longer coherent output is better. Failed OCR often produces truncated or empty results. Normalized against page count. |

**Decision logic**:
- Both scores above quality_threshold (0.6): pick higher score
- One above, one below: pick the one above
- Both below threshold: return higher score but set `quality_warning=True`

## Data Model Changes

### LoadedDocument (updated)

```python
@dataclass
class LoadedDocument:
    text: str
    source_path: Path
    format: str
    page_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    ocr_engine_used: Optional[str] = None       # "surya", "chandra", or None
    ocr_confidence: Optional[float] = None       # quality score 0-1
    quality_warning: bool = False                 # True if below threshold
    pages: list[PageResult] = field(default_factory=list)
```

### PageResult (new)

```python
@dataclass
class PageResult:
    page_number: int
    text: str
    engine: str              # which engine produced this
    confidence: float        # quality score
    layout_html: Optional[str] = None  # Chandra HTML with bounding boxes
```

Per-page scoring allows mixed documents (typed + handwritten pages) to use the best engine for each page.

## Configuration

New and updated fields in MosaicxConfig:

```python
# --- Document loading ---
ocr_engine: Literal["both", "surya", "chandra"] = "both"
chandra_backend: Literal["vllm", "hf", "auto"] = "auto"
chandra_server_url: str = ""            # optional: connect to running vLLM server
quality_threshold: float = 0.6
ocr_page_timeout: int = 60             # seconds per page
force_ocr: bool = False                 # run OCR even on native PDFs
ocr_langs: list[str] = ["en", "de"]    # passed to Surya
```

Chandra backend auto-detection:
- CUDA available -> vLLM (highest throughput on NVIDIA)
- MPS available -> HuggingFace transformers (Mac Metal acceleration)

## Module Structure

```
mosaicx/documents/
    __init__.py              # re-exports load_document, LoadedDocument
    loader.py                # main load_document() orchestrator
    models.py                # LoadedDocument, PageResult dataclasses
    quality.py               # QualityScorer
    engines/
        __init__.py
        base.py              # OCREngine protocol
        surya_engine.py      # Surya wrapper
        chandra_engine.py    # Chandra wrapper (HF + vLLM)
```

Engine protocol:

```python
class OCREngine(Protocol):
    def ocr_pages(self, images: list[Image]) -> list[PageResult]: ...
```

## Dependencies

```toml
dependencies = [
    # ... existing deps ...
    "surya-ocr>=0.17.0",
    "chandra-ocr>=1.0.0",
]
```

Remove `docling>=2.0.0`. Both Surya and Chandra are core dependencies (accuracy-first, users have GPUs).

New supported formats: `.jpg`, `.png`, `.tiff` (image files) in addition to existing `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Chandra server/model unreachable | Warn, fall back to Surya only |
| Surya fails on a page | Use Chandra result for that page |
| Both engines fail on a page | Empty text, `quality_warning=True` |
| Both scores below threshold | Return best result, `quality_warning=True` |
| Empty PDF (0 pages) | Return empty LoadedDocument, `is_empty=True` |
| Password-protected PDF | Raise `DocumentLoadError` |
| Corrupted file | Raise `DocumentLoadError` |
| Page timeout (>60s) | Use other engine's result |
| `.txt`/`.md` files | Skip OCR entirely, direct read |

New exception: `DocumentLoadError` replaces the current mix of FileNotFoundError/ValueError/RuntimeError.

## Testing Strategy

**Unit tests** (mock OCR engines, no GPU):
- Text files skip OCR
- Quality scorer picks higher score
- Quality scorer flags warning when both below threshold
- Per-page winner selection for mixed documents
- Medical vocab scoring accuracy
- Character sanity scoring
- Graceful fallback when one engine fails
- Page timeout handling
- DocumentLoadError for corrupted files
- Engine protocol compliance

**Integration tests** (`@pytest.mark.slow`, need GPU):
- Native PDF roundtrip
- Scanned typed PDF extraction
- Image file OCR
- Dual engine produces output for same input

## Target Hardware

- Mac Studio (64-512GB unified memory) — Surya + Chandra via HF/MPS
- NVIDIA Digit Spark (128GB unified memory) — Surya + Chandra via vLLM/CUDA
- NVIDIA RTX 6000 (96GB VRAM) — Surya + Chandra via vLLM/CUDA

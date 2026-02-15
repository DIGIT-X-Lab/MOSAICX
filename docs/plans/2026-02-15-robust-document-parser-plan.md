# Robust Document Parser Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Docling with a dual-engine OCR system (Surya + Chandra) that runs both in parallel, quality-scores per page, and picks the best result — accuracy-first for medical documents.

**Architecture:** Every PDF/image document goes through both Surya (fast, layout-aware OCR) and Chandra (VLM-based, handles handwriting/forms). A deterministic quality scorer compares per-page output and picks the winner. Text files skip OCR entirely.

**Tech Stack:** surya-ocr, chandra-ocr, pypdfium2 (PDF→images), Pillow, ThreadPoolExecutor

---

### Task 1: Update dependencies in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update dependencies**

Replace `docling>=2.0.0` with:
```toml
dependencies = [
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "dspy>=2.6.0",
    "surya-ocr>=0.17.0",
    "chandra-ocr>=1.0.0",
    "pypdfium2>=4.0.0",
    "Pillow>=10.0.0",
    "pyyaml>=6.0",
    "typing-extensions>=4.8.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
]
```

Remove the `pdf` and `docx` optional dependency groups (no longer needed since Docling handled those). Keep `hf` and `dev`.

**Step 2: Install updated dependencies**

Run: `uv sync`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: replace docling with surya-ocr + chandra-ocr + pypdfium2"
```

---

### Task 2: Data models (PageResult, LoadedDocument, DocumentLoadError)

**Files:**
- Create: `mosaicx/documents/models.py`
- Modify: `mosaicx/documents/loader.py` — remove LoadedDocument (moved to models.py)
- Modify: `mosaicx/documents/__init__.py` — update imports
- Create: `tests/test_document_models.py`

**Step 1: Write the failing test**

```python
# tests/test_document_models.py
"""Tests for document data models."""

import pytest
from pathlib import Path


class TestPageResult:
    def test_construction(self):
        from mosaicx.documents.models import PageResult

        page = PageResult(
            page_number=1,
            text="Patient presents with cough.",
            engine="surya",
            confidence=0.85,
        )
        assert page.page_number == 1
        assert page.engine == "surya"
        assert page.confidence == 0.85
        assert page.layout_html is None

    def test_with_layout_html(self):
        from mosaicx.documents.models import PageResult

        page = PageResult(
            page_number=1,
            text="Test",
            engine="chandra",
            confidence=0.9,
            layout_html="<div>Test</div>",
        )
        assert page.layout_html == "<div>Test</div>"


class TestLoadedDocumentNewFields:
    def test_ocr_metadata_defaults(self):
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(
            text="Hello",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
        )
        assert doc.ocr_engine_used is None
        assert doc.ocr_confidence is None
        assert doc.quality_warning is False
        assert doc.pages == []

    def test_ocr_metadata_populated(self):
        from mosaicx.documents.models import LoadedDocument, PageResult

        pages = [
            PageResult(page_number=1, text="Page 1", engine="surya", confidence=0.9),
            PageResult(page_number=2, text="Page 2", engine="chandra", confidence=0.7),
        ]
        doc = LoadedDocument(
            text="Page 1\nPage 2",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            page_count=2,
            ocr_engine_used="mixed",
            ocr_confidence=0.8,
            quality_warning=False,
            pages=pages,
        )
        assert doc.ocr_engine_used == "mixed"
        assert len(doc.pages) == 2
        assert doc.pages[0].engine == "surya"
        assert doc.pages[1].engine == "chandra"

    def test_backward_compat_char_count_is_empty(self):
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(text="abc", source_path=Path("/tmp/x.txt"), format="txt")
        assert doc.char_count == 3
        assert doc.is_empty is False

        empty = LoadedDocument(text="", source_path=Path("/tmp/x.txt"), format="txt")
        assert empty.is_empty is True


class TestDocumentLoadError:
    def test_is_exception(self):
        from mosaicx.documents.models import DocumentLoadError

        err = DocumentLoadError("Corrupted PDF")
        assert isinstance(err, Exception)
        assert str(err) == "Corrupted PDF"
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_document_models.py -x -v`
Expected: FAIL (models.py doesn't exist)

**Step 3: Implement models.py**

```python
# mosaicx/documents/models.py
"""Document data models for the loading pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


@dataclass
class PageResult:
    """OCR result for a single page."""

    page_number: int
    text: str
    engine: str
    confidence: float
    layout_html: Optional[str] = None


@dataclass
class LoadedDocument:
    """A document converted to plain text with OCR metadata."""

    text: str
    source_path: Path
    format: str
    page_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    ocr_engine_used: Optional[str] = None
    ocr_confidence: Optional[float] = None
    quality_warning: bool = False
    pages: list[PageResult] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0
```

Then update `mosaicx/documents/loader.py` — remove the `LoadedDocument` dataclass from it and import from models instead. Update `mosaicx/documents/__init__.py` to re-export from models:

```python
# mosaicx/documents/__init__.py
"""Document loading — dual-engine OCR (Surya + Chandra) + plain text."""
from .models import LoadedDocument, PageResult, DocumentLoadError
from .loader import load_document

__all__ = ["LoadedDocument", "PageResult", "DocumentLoadError", "load_document"]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_document_models.py tests/test_documents.py -x -v`
Expected: All pass (new tests + existing tests still work via backward compat)

**Step 5: Commit**

```bash
git add mosaicx/documents/models.py mosaicx/documents/__init__.py mosaicx/documents/loader.py tests/test_document_models.py
git commit -m "feat: add PageResult, DocumentLoadError, extend LoadedDocument with OCR metadata"
```

---

### Task 3: Quality scorer

**Files:**
- Create: `mosaicx/documents/quality.py`
- Create: `tests/test_quality_scorer.py`

**Step 1: Write the failing test**

```python
# tests/test_quality_scorer.py
"""Tests for the OCR quality scorer."""

import pytest


class TestMedicalVocabScore:
    def test_medical_text_scores_high(self):
        from mosaicx.documents.quality import medical_vocab_score

        text = "CT chest shows 5mm nodule in the right upper lobe. No pleural effusion. Heart is normal size."
        score = medical_vocab_score(text)
        assert score > 0.3

    def test_garbled_text_scores_low(self):
        from mosaicx.documents.quality import medical_vocab_score

        text = "Th3 p@t13nt pr3s3nts w1th c0ugh. Xr@y sh0ws n0 @bn0rm@l1ty."
        score = medical_vocab_score(text)
        assert score < 0.1

    def test_empty_text_scores_zero(self):
        from mosaicx.documents.quality import medical_vocab_score

        assert medical_vocab_score("") == 0.0


class TestCharSanityScore:
    def test_clean_text_scores_high(self):
        from mosaicx.documents.quality import char_sanity_score

        text = "The lungs are clear bilaterally. No consolidation or effusion."
        score = char_sanity_score(text)
        assert score > 0.8

    def test_garbled_text_scores_low(self):
        from mosaicx.documents.quality import char_sanity_score

        text = "T#3 l@ng$ @r3 cl3@r b!l@t3r@lly. N0 c0ns0l!d@t!0n."
        score = char_sanity_score(text)
        assert score < 0.7

    def test_empty_text_scores_zero(self):
        from mosaicx.documents.quality import char_sanity_score

        assert char_sanity_score("") == 0.0


class TestWordStructureScore:
    def test_normal_text_scores_high(self):
        from mosaicx.documents.quality import word_structure_score

        text = "Patient presents with persistent cough for three weeks."
        score = word_structure_score(text)
        assert score > 0.7

    def test_fragmented_text_scores_low(self):
        from mosaicx.documents.quality import word_structure_score

        text = "P a t i e n t p r e s e n t s w i t h c o u g h"
        score = word_structure_score(text)
        assert score < 0.5

    def test_empty_text_scores_zero(self):
        from mosaicx.documents.quality import word_structure_score

        assert word_structure_score("") == 0.0


class TestQualityScorer:
    def test_good_medical_text(self):
        from mosaicx.documents.quality import QualityScorer

        scorer = QualityScorer()
        text = (
            "CT chest with contrast. Indication: cough.\n"
            "Findings: 5mm ground glass nodule in the right upper lobe.\n"
            "No pleural effusion. Heart is normal size.\n"
            "Impression: Pulmonary nodule, recommend follow-up."
        )
        score = scorer.score(text)
        assert score > 0.5

    def test_garbled_text_low_score(self):
        from mosaicx.documents.quality import QualityScorer

        scorer = QualityScorer()
        text = "@#$ %^& *() !@# $%^ &*( )!@ #$% ^&*"
        score = scorer.score(text)
        assert score < 0.3

    def test_empty_text_zero(self):
        from mosaicx.documents.quality import QualityScorer

        scorer = QualityScorer()
        assert scorer.score("") == 0.0

    def test_score_in_range(self):
        from mosaicx.documents.quality import QualityScorer

        scorer = QualityScorer()
        for text in ["hello world", "CT scan normal", "", "@@@@"]:
            score = scorer.score(text)
            assert 0.0 <= score <= 1.0
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_quality_scorer.py -x -v`
Expected: FAIL (quality.py doesn't exist)

**Step 3: Implement quality.py**

```python
# mosaicx/documents/quality.py
"""Deterministic quality scorer for OCR output.

Evaluates text quality using four metrics: medical vocabulary hit rate,
character sanity, word structure, and text length. No LLM dependency.
"""

from __future__ import annotations

import re
import string

# Curated medical vocabulary (~200 terms). Covers anatomy, findings,
# modalities, procedures, and common clinical language. Lowercase for
# case-insensitive matching.
MEDICAL_VOCAB: frozenset[str] = frozenset({
    # Anatomy - thorax
    "lung", "lungs", "lobe", "lobes", "bronchus", "bronchi", "trachea",
    "mediastinum", "pleura", "pleural", "diaphragm", "hilum", "hilar",
    "pericardium", "pericardial", "cardiac", "heart", "aorta", "aortic",
    "pulmonary", "thorax", "thoracic", "sternum", "rib", "ribs",
    # Anatomy - abdomen
    "liver", "hepatic", "spleen", "splenic", "kidney", "kidneys", "renal",
    "pancreas", "pancreatic", "gallbladder", "adrenal", "bowel", "colon",
    "intestine", "stomach", "gastric", "duodenum", "appendix", "mesentery",
    "peritoneum", "peritoneal", "retroperitoneal", "bladder", "uterus",
    "ovary", "prostate",
    # Anatomy - neuro
    "brain", "cerebral", "cerebellum", "cerebellar", "ventricle",
    "ventricular", "cortex", "cortical", "hippocampus", "thalamus",
    "brainstem", "pons", "midbrain", "skull", "calvarium", "meninges",
    "meningeal", "dura", "dural", "spinal", "spine", "vertebra",
    "vertebral", "disc", "foramen",
    # Anatomy - MSK
    "femur", "tibia", "fibula", "humerus", "radius", "ulna", "pelvis",
    "acetabulum", "scapula", "clavicle", "patella", "meniscus", "tendon",
    "ligament", "cartilage", "joint", "muscle", "bone", "marrow",
    # Anatomy - vascular
    "artery", "arterial", "vein", "venous", "vessel", "vascular",
    "carotid", "jugular", "subclavian", "iliac", "femoral", "portal",
    # Findings / pathology
    "nodule", "nodular", "mass", "lesion", "tumor", "tumour", "cyst",
    "calcification", "opacity", "opacification", "consolidation",
    "atelectasis", "effusion", "edema", "oedema", "hemorrhage",
    "haemorrhage", "infarct", "infarction", "thrombosis", "thrombus",
    "embolism", "stenosis", "occlusion", "aneurysm", "dissection",
    "fracture", "dislocation", "erosion", "sclerosis", "fibrosis",
    "necrosis", "abscess", "inflammation", "metastasis", "metastatic",
    "lymphadenopathy", "cardiomegaly", "hepatomegaly", "splenomegaly",
    "pneumothorax", "pneumonia", "emphysema", "hernia", "herniation",
    "hydrocephalus", "demyelination", "enhancement", "enhancing",
    # Modalities / procedures
    "ct", "mri", "mra", "pet", "spect", "ultrasound", "radiograph",
    "x-ray", "xray", "fluoroscopy", "mammography", "angiography",
    "echocardiography", "doppler", "scan", "imaging", "contrast",
    "gadolinium", "iodine",
    # Clinical terms
    "patient", "diagnosis", "impression", "findings", "indication",
    "history", "clinical", "chronic", "acute", "bilateral", "unilateral",
    "anterior", "posterior", "lateral", "medial", "superior", "inferior",
    "proximal", "distal", "diffuse", "focal", "mild", "moderate",
    "severe", "benign", "malignant", "normal", "abnormal", "stable",
    "unchanged", "interval", "follow-up", "followup", "prior",
    "comparison", "technique", "protocol", "axial", "coronal", "sagittal",
    # Measurements / descriptors
    "mm", "cm", "diameter", "size", "volume", "density", "signal",
    "intensity", "attenuation", "hyperintense", "hypointense",
    "hyperdense", "hypodense", "heterogeneous", "homogeneous",
    "circumscribed", "irregular", "spiculated", "ground-glass",
})

# Characters considered "sane" in medical text
_SANE_CHARS = set(string.ascii_letters + string.digits + string.whitespace + ".,;:!?'-/()+=%°")


def medical_vocab_score(text: str) -> float:
    """Score 0-1 based on fraction of words matching medical vocabulary."""
    if not text.strip():
        return 0.0
    words = re.findall(r"[a-zA-Z][\w'-]*", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in MEDICAL_VOCAB)
    # Normalize: a good medical report has ~15-30% medical terms
    raw_ratio = hits / len(words)
    # Scale so that 0.15 ratio -> 0.7 score, 0.30 -> 1.0
    return min(1.0, raw_ratio / 0.30)


def char_sanity_score(text: str) -> float:
    """Score 0-1 based on fraction of characters that are 'normal'."""
    if not text:
        return 0.0
    sane = sum(1 for c in text if c in _SANE_CHARS)
    return sane / len(text)


def word_structure_score(text: str) -> float:
    """Score 0-1 based on word length distribution and whitespace ratio."""
    if not text.strip():
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    # Ideal average word length for English medical text: 4-8 chars
    if 3.0 <= avg_len <= 10.0:
        len_score = 1.0
    elif avg_len < 2.0 or avg_len > 15.0:
        len_score = 0.2
    else:
        len_score = 0.6
    # Whitespace ratio: should be ~15-25% of total chars
    ws_ratio = sum(1 for c in text if c in " \t\n") / len(text) if text else 0
    if 0.10 <= ws_ratio <= 0.35:
        ws_score = 1.0
    elif ws_ratio < 0.05 or ws_ratio > 0.50:
        ws_score = 0.2
    else:
        ws_score = 0.6
    return (len_score + ws_score) / 2.0


def text_length_score(text: str, expected_pages: int = 1) -> float:
    """Score 0-1 based on text length relative to page count.

    A typical medical report page has ~300-800 words.
    """
    if not text.strip():
        return 0.0
    word_count = len(text.split())
    expected_words = expected_pages * 400  # midpoint estimate
    if expected_words == 0:
        return 0.5
    ratio = word_count / expected_words
    # 0.5-2.0x expected is good, outside that degrades
    if 0.3 <= ratio <= 2.5:
        return 1.0
    elif ratio < 0.1:
        return 0.1
    else:
        return 0.5


class QualityScorer:
    """Deterministic scorer for OCR output quality.

    Combines medical vocabulary, character sanity, word structure,
    and text length into a single 0-1 score.
    """

    WEIGHTS = {
        "medical_vocab": 0.35,
        "char_sanity": 0.25,
        "word_structure": 0.20,
        "text_length": 0.20,
    }

    def score(self, text: str, expected_pages: int = 1) -> float:
        """Return overall quality score 0-1."""
        if not text.strip():
            return 0.0
        return (
            self.WEIGHTS["medical_vocab"] * medical_vocab_score(text)
            + self.WEIGHTS["char_sanity"] * char_sanity_score(text)
            + self.WEIGHTS["word_structure"] * word_structure_score(text)
            + self.WEIGHTS["text_length"] * text_length_score(text, expected_pages)
        )

    def score_detailed(self, text: str, expected_pages: int = 1) -> dict[str, float]:
        """Return per-metric scores plus overall."""
        detail = {
            "medical_vocab": medical_vocab_score(text),
            "char_sanity": char_sanity_score(text),
            "word_structure": word_structure_score(text),
            "text_length": text_length_score(text, expected_pages),
        }
        detail["overall"] = sum(
            self.WEIGHTS[k] * v for k, v in detail.items()
        )
        return detail
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_quality_scorer.py -x -v`
Expected: All pass

**Step 5: Commit**

```bash
git add mosaicx/documents/quality.py tests/test_quality_scorer.py
git commit -m "feat: add deterministic OCR quality scorer with medical vocabulary"
```

---

### Task 4: OCR engine protocol and image utilities

**Files:**
- Create: `mosaicx/documents/engines/__init__.py`
- Create: `mosaicx/documents/engines/base.py`
- Create: `tests/test_engine_base.py`

**Step 1: Write the failing test**

```python
# tests/test_engine_base.py
"""Tests for OCR engine protocol and image utilities."""

import pytest
from pathlib import Path


class TestOCREngineProtocol:
    def test_protocol_is_importable(self):
        from mosaicx.documents.engines.base import OCREngine
        assert OCREngine is not None

    def test_protocol_defines_ocr_pages(self):
        from mosaicx.documents.engines.base import OCREngine
        import inspect
        members = dict(inspect.getmembers(OCREngine))
        assert "ocr_pages" in members or hasattr(OCREngine, "ocr_pages")


class TestPdfToImages:
    def test_function_exists(self):
        from mosaicx.documents.engines.base import pdf_to_images
        assert callable(pdf_to_images)

    def test_image_to_pages(self, tmp_path):
        from mosaicx.documents.engines.base import image_to_pages
        from PIL import Image

        # Create a small test image
        img = Image.new("RGB", (100, 100), color="white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        pages = image_to_pages(img_path)
        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)


class TestSupportedFormats:
    def test_supported_formats_exported(self):
        from mosaicx.documents.engines.base import SUPPORTED_FORMATS
        assert ".pdf" in SUPPORTED_FORMATS
        assert ".jpg" in SUPPORTED_FORMATS
        assert ".png" in SUPPORTED_FORMATS
        assert ".tiff" in SUPPORTED_FORMATS
        assert ".txt" in SUPPORTED_FORMATS
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_engine_base.py -x -v`
Expected: FAIL

**Step 3: Implement**

```python
# mosaicx/documents/engines/__init__.py
"""OCR engine implementations."""
```

```python
# mosaicx/documents/engines/base.py
"""OCR engine protocol and shared utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image

from ..models import PageResult

# All formats the loader accepts
TEXT_FORMATS = frozenset({".txt", ".md", ".markdown"})
IMAGE_FORMATS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"})
PDF_FORMATS = frozenset({".pdf"})
DOCLING_FORMATS = frozenset({".docx", ".pptx"})  # still text-extractable
SUPPORTED_FORMATS = TEXT_FORMATS | IMAGE_FORMATS | PDF_FORMATS | DOCLING_FORMATS


@runtime_checkable
class OCREngine(Protocol):
    """Protocol for OCR engines."""

    def ocr_pages(self, images: list[Image.Image], langs: list[str] | None = None) -> list[PageResult]:
        """Run OCR on a list of page images.

        Parameters
        ----------
        images : list of PIL Images, one per page.
        langs  : language hints (e.g. ["en", "de"]).

        Returns
        -------
        list[PageResult] with one entry per input image.
        """
        ...


def pdf_to_images(path: Path, dpi: int = 200) -> list[Image.Image]:
    """Convert a PDF file to a list of PIL Images (one per page).

    Uses pypdfium2 — no system dependencies (unlike poppler).
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=dpi / 72)
        images.append(bitmap.to_pil())
    pdf.close()
    return images


def image_to_pages(path: Path) -> list[Image.Image]:
    """Load an image file as a single-page list."""
    img = Image.open(path).convert("RGB")
    return [img]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_engine_base.py -x -v`
Expected: All pass

**Step 5: Commit**

```bash
git add mosaicx/documents/engines/ tests/test_engine_base.py
git commit -m "feat: add OCR engine protocol, pdf_to_images, and image utilities"
```

---

### Task 5: Surya engine wrapper

**Files:**
- Create: `mosaicx/documents/engines/surya_engine.py`
- Create: `tests/test_surya_engine.py`

**Step 1: Write the failing test**

```python
# tests/test_surya_engine.py
"""Tests for the Surya OCR engine wrapper."""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from mosaicx.documents.models import PageResult


class TestSuryaEngine:
    def test_implements_protocol(self):
        from mosaicx.documents.engines.base import OCREngine
        from mosaicx.documents.engines.surya_engine import SuryaEngine

        engine = SuryaEngine.__new__(SuryaEngine)
        assert isinstance(engine, OCREngine)

    def test_ocr_pages_returns_page_results(self):
        from mosaicx.documents.engines.surya_engine import SuryaEngine

        # Mock surya internals
        with patch("mosaicx.documents.engines.surya_engine._get_surya_pipeline") as mock_pipeline:
            mock_result = MagicMock()
            mock_result.text_lines = [
                MagicMock(text="Patient presents with cough."),
                MagicMock(text="CT shows 5mm nodule."),
            ]
            mock_result.confidence = 0.92
            mock_pipeline.return_value.return_value = [mock_result]

            engine = SuryaEngine()
            images = [Image.new("RGB", (100, 100), "white")]
            results = engine.ocr_pages(images, langs=["en"])

            assert len(results) == 1
            assert isinstance(results[0], PageResult)
            assert results[0].engine == "surya"
            assert "Patient presents" in results[0].text

    def test_empty_image_list(self):
        from mosaicx.documents.engines.surya_engine import SuryaEngine

        with patch("mosaicx.documents.engines.surya_engine._get_surya_pipeline"):
            engine = SuryaEngine()
            results = engine.ocr_pages([], langs=["en"])
            assert results == []
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_surya_engine.py -x -v`
Expected: FAIL

**Step 3: Implement**

```python
# mosaicx/documents/engines/surya_engine.py
"""Surya OCR engine wrapper.

Surya provides fast layout-aware OCR using traditional pipeline
(detection -> recognition). Works on CPU and GPU (CUDA/MPS).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from PIL import Image

from ..models import PageResult

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_surya_pipeline():
    """Lazily load and cache the Surya OCR pipeline."""
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    def run_ocr(images: list[Image.Image], langs: list[str] | None = None):
        from surya.pipeline import OCRPipeline
        pipeline = OCRPipeline(det_predictor=det_predictor, rec_predictor=rec_predictor)
        return pipeline(images, langs=langs or ["en"])

    return run_ocr


class SuryaEngine:
    """OCR engine backed by Surya."""

    def ocr_pages(
        self,
        images: list[Image.Image],
        langs: list[str] | None = None,
    ) -> list[PageResult]:
        if not images:
            return []

        pipeline = _get_surya_pipeline()
        try:
            results = pipeline(images, langs=langs)
        except Exception:
            logger.exception("Surya OCR failed")
            return [
                PageResult(page_number=i + 1, text="", engine="surya", confidence=0.0)
                for i in range(len(images))
            ]

        page_results = []
        for i, result in enumerate(results):
            lines = [line.text for line in result.text_lines]
            text = "\n".join(lines)
            conf = getattr(result, "confidence", 0.8)
            page_results.append(
                PageResult(
                    page_number=i + 1,
                    text=text,
                    engine="surya",
                    confidence=float(conf),
                )
            )
        return page_results
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_surya_engine.py -x -v`
Expected: All pass

**Step 5: Commit**

```bash
git add mosaicx/documents/engines/surya_engine.py tests/test_surya_engine.py
git commit -m "feat: add Surya OCR engine wrapper"
```

---

### Task 6: Chandra engine wrapper

**Files:**
- Create: `mosaicx/documents/engines/chandra_engine.py`
- Create: `tests/test_chandra_engine.py`

**Step 1: Write the failing test**

```python
# tests/test_chandra_engine.py
"""Tests for the Chandra OCR engine wrapper."""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from mosaicx.documents.models import PageResult


class TestChandraEngine:
    def test_implements_protocol(self):
        from mosaicx.documents.engines.base import OCREngine
        from mosaicx.documents.engines.chandra_engine import ChandraEngine

        engine = ChandraEngine.__new__(ChandraEngine)
        assert isinstance(engine, OCREngine)

    def test_ocr_pages_returns_page_results(self):
        from mosaicx.documents.engines.chandra_engine import ChandraEngine

        with patch("mosaicx.documents.engines.chandra_engine._get_chandra_manager") as mock_mgr:
            mock_result = MagicMock()
            mock_result.markdown = "Patient presents with cough.\n\n5mm nodule in RUL."
            mock_result.html = "<div>Patient presents with cough.</div>"
            mock_mgr.return_value.generate.return_value = [mock_result]

            engine = ChandraEngine()
            images = [Image.new("RGB", (100, 100), "white")]
            results = engine.ocr_pages(images)

            assert len(results) == 1
            assert isinstance(results[0], PageResult)
            assert results[0].engine == "chandra"
            assert "Patient presents" in results[0].text
            assert results[0].layout_html is not None

    def test_auto_backend_detection(self):
        from mosaicx.documents.engines.chandra_engine import _detect_backend

        backend = _detect_backend()
        assert backend in ("vllm", "hf")

    def test_empty_image_list(self):
        from mosaicx.documents.engines.chandra_engine import ChandraEngine

        with patch("mosaicx.documents.engines.chandra_engine._get_chandra_manager"):
            engine = ChandraEngine()
            results = engine.ocr_pages([])
            assert results == []
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chandra_engine.py -x -v`
Expected: FAIL

**Step 3: Implement**

```python
# mosaicx/documents/engines/chandra_engine.py
"""Chandra OCR engine wrapper.

Chandra is a VLM-based OCR (fine-tuned Qwen3-VL 9B) that handles
handwriting, complex forms, tables, and mixed layouts. Supports
vLLM (CUDA) and HuggingFace transformers (CUDA/MPS) backends.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch
from PIL import Image

from ..models import PageResult

logger = logging.getLogger(__name__)


def _detect_backend() -> str:
    """Auto-detect best backend: vLLM on CUDA, HF on MPS/CPU."""
    if torch.cuda.is_available():
        return "vllm"
    return "hf"


@lru_cache(maxsize=1)
def _get_chandra_manager(backend: str | None = None):
    """Lazily load and cache the Chandra inference manager."""
    from chandra.model import InferenceManager

    method = backend or _detect_backend()
    return InferenceManager(method=method)


class ChandraEngine:
    """OCR engine backed by Chandra VLM."""

    def __init__(self, backend: str | None = None, server_url: str | None = None):
        self._backend = backend
        self._server_url = server_url

    def ocr_pages(
        self,
        images: list[Image.Image],
        langs: list[str] | None = None,
    ) -> list[PageResult]:
        if not images:
            return []

        manager = _get_chandra_manager(self._backend)
        try:
            results = manager.generate(images)
        except Exception:
            logger.exception("Chandra OCR failed")
            return [
                PageResult(page_number=i + 1, text="", engine="chandra", confidence=0.0)
                for i in range(len(images))
            ]

        page_results = []
        for i, result in enumerate(results):
            text = getattr(result, "markdown", "") or ""
            html = getattr(result, "html", None)
            page_results.append(
                PageResult(
                    page_number=i + 1,
                    text=text,
                    engine="chandra",
                    confidence=0.9,  # Chandra doesn't expose per-page confidence
                    layout_html=html,
                )
            )
        return page_results
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_chandra_engine.py -x -v`
Expected: All pass

**Step 5: Commit**

```bash
git add mosaicx/documents/engines/chandra_engine.py tests/test_chandra_engine.py
git commit -m "feat: add Chandra VLM OCR engine wrapper with auto backend detection"
```

---

### Task 7: Rewrite loader orchestrator

**Files:**
- Modify: `mosaicx/documents/loader.py` — full rewrite
- Create: `tests/test_loader_orchestrator.py`

**Step 1: Write the failing test**

```python
# tests/test_loader_orchestrator.py
"""Tests for the document loader orchestrator (dual-engine)."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from mosaicx.documents.models import PageResult, LoadedDocument, DocumentLoadError


def _make_page(page_num, text, engine, confidence):
    return PageResult(page_number=page_num, text=text, engine=engine, confidence=confidence)


class TestTextFileBypass:
    """Text files should skip OCR entirely."""

    def test_txt_no_ocr(self, tmp_path):
        from mosaicx.documents.loader import load_document

        f = tmp_path / "report.txt"
        f.write_text("Patient presents with cough.")
        doc = load_document(f)
        assert doc.ocr_engine_used is None
        assert doc.pages == []
        assert "Patient presents" in doc.text

    def test_md_no_ocr(self, tmp_path):
        from mosaicx.documents.loader import load_document

        f = tmp_path / "report.md"
        f.write_text("# Findings\nNormal.")
        doc = load_document(f)
        assert doc.ocr_engine_used is None


class TestDualEngineOrchestration:
    """Test the parallel engine dispatch and quality scoring."""

    def test_picks_higher_quality_engine(self, tmp_path):
        from mosaicx.documents.loader import _pick_best_pages

        surya_pages = [_make_page(1, "Good medical text with nodule", "surya", 0.8)]
        chandra_pages = [_make_page(1, "G00d m3d1c@l t3xt", "chandra", 0.3)]

        winners = _pick_best_pages(surya_pages, chandra_pages)
        assert len(winners) == 1
        assert winners[0].engine == "surya"

    def test_mixed_page_winners(self):
        from mosaicx.documents.loader import _pick_best_pages

        surya_pages = [
            _make_page(1, "Good typed text with findings and impression", "surya", 0.9),
            _make_page(2, "@#$ garbled", "surya", 0.1),
        ]
        chandra_pages = [
            _make_page(1, "OK text", "chandra", 0.5),
            _make_page(2, "Handwritten note about patient diagnosis", "chandra", 0.8),
        ]

        winners = _pick_best_pages(surya_pages, chandra_pages)
        assert winners[0].engine == "surya"   # page 1: surya better
        assert winners[1].engine == "chandra" # page 2: chandra better

    def test_quality_warning_when_both_low(self):
        from mosaicx.documents.loader import _pick_best_pages, _assemble_document

        surya_pages = [_make_page(1, "@@@", "surya", 0.1)]
        chandra_pages = [_make_page(1, "###", "chandra", 0.1)]

        winners = _pick_best_pages(surya_pages, chandra_pages, threshold=0.6)
        doc = _assemble_document(
            winners=winners,
            source_path=Path("/tmp/test.pdf"),
            fmt="pdf",
            threshold=0.6,
        )
        assert doc.quality_warning is True


class TestErrorHandling:
    def test_missing_file_raises(self):
        from mosaicx.documents.loader import load_document
        from mosaicx.documents.models import DocumentLoadError

        with pytest.raises((FileNotFoundError, DocumentLoadError)):
            load_document(Path("/nonexistent/file.pdf"))

    def test_unsupported_format_raises(self, tmp_path):
        from mosaicx.documents.loader import load_document

        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(f)


class TestImageFormats:
    def test_image_formats_accepted(self):
        from mosaicx.documents.engines.base import SUPPORTED_FORMATS

        for ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
            assert ext in SUPPORTED_FORMATS
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_loader_orchestrator.py -x -v`
Expected: Partial fail (text file tests may pass, new functions don't exist)

**Step 3: Rewrite loader.py**

```python
# mosaicx/documents/loader.py
"""Document loading orchestrator — dual-engine OCR (Surya + Chandra).

Routes documents through the appropriate loading path:
- .txt/.md: direct read (no OCR)
- .pdf/.jpg/.png/.tiff: parallel OCR with Surya + Chandra, quality scoring
- .docx/.pptx: text extraction (if docling available), else error
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .engines.base import (
    IMAGE_FORMATS,
    PDF_FORMATS,
    SUPPORTED_FORMATS,
    TEXT_FORMATS,
    image_to_pages,
    pdf_to_images,
)
from .models import DocumentLoadError, LoadedDocument, PageResult
from .quality import QualityScorer

logger = logging.getLogger(__name__)

_scorer = QualityScorer()


def load_document(
    path: Path,
    ocr_engine: str = "both",
    force_ocr: bool = False,
    ocr_langs: list[str] | None = None,
    chandra_backend: str | None = None,
    quality_threshold: float = 0.6,
    page_timeout: int = 60,
) -> LoadedDocument:
    """Load a document from disk with automatic OCR if needed.

    Parameters
    ----------
    path            : Path to the document.
    ocr_engine      : "both", "surya", or "chandra".
    force_ocr       : Run OCR even on native PDFs with text layers.
    ocr_langs       : Language hints for Surya (e.g. ["en", "de"]).
    chandra_backend : "vllm", "hf", or None for auto-detect.
    quality_threshold : Minimum quality score before flagging warning.
    page_timeout    : Max seconds per page for each engine.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    # Text files: direct read, no OCR
    if suffix in TEXT_FORMATS:
        return _load_text(path, suffix.lstrip("."))

    # PDF and image files: OCR pipeline
    if suffix in PDF_FORMATS:
        images = pdf_to_images(path)
    elif suffix in IMAGE_FORMATS:
        images = image_to_pages(path)
    else:
        raise DocumentLoadError(f"Cannot process format: {suffix}")

    if not images:
        return LoadedDocument(
            text="", source_path=path, format=suffix.lstrip("."),
            page_count=0, quality_warning=True,
        )

    # Run OCR engines
    surya_pages, chandra_pages = _run_engines(
        images=images,
        ocr_engine=ocr_engine,
        ocr_langs=ocr_langs or ["en"],
        chandra_backend=chandra_backend,
        page_timeout=page_timeout,
    )

    # Pick best per page
    winners = _pick_best_pages(surya_pages, chandra_pages, threshold=quality_threshold)

    return _assemble_document(
        winners=winners,
        source_path=path,
        fmt=suffix.lstrip("."),
        threshold=quality_threshold,
    )


def _load_text(path: Path, fmt: str) -> LoadedDocument:
    """Load a plain text file."""
    text = path.read_text(encoding="utf-8")
    return LoadedDocument(text=text, source_path=path, format=fmt)


def _run_engines(
    images: list,
    ocr_engine: str,
    ocr_langs: list[str],
    chandra_backend: str | None,
    page_timeout: int,
) -> tuple[list[PageResult], list[PageResult]]:
    """Dispatch OCR engines in parallel and collect results."""
    surya_pages: list[PageResult] = []
    chandra_pages: list[PageResult] = []

    def run_surya():
        from .engines.surya_engine import SuryaEngine
        engine = SuryaEngine()
        return engine.ocr_pages(images, langs=ocr_langs)

    def run_chandra():
        from .engines.chandra_engine import ChandraEngine
        engine = ChandraEngine(backend=chandra_backend)
        return engine.ocr_pages(images)

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if ocr_engine in ("both", "surya"):
            futures["surya"] = pool.submit(run_surya)
        if ocr_engine in ("both", "chandra"):
            futures["chandra"] = pool.submit(run_chandra)

        for name, future in futures.items():
            try:
                result = future.result(timeout=page_timeout * len(images))
            except Exception:
                logger.warning("OCR engine '%s' failed, skipping", name)
                empty = [
                    PageResult(page_number=i + 1, text="", engine=name, confidence=0.0)
                    for i in range(len(images))
                ]
                result = empty

            if name == "surya":
                surya_pages = result
            else:
                chandra_pages = result

    return surya_pages, chandra_pages


def _pick_best_pages(
    surya_pages: list[PageResult],
    chandra_pages: list[PageResult],
    threshold: float = 0.6,
) -> list[PageResult]:
    """Compare per-page quality scores and pick the best engine for each page."""
    # Handle single-engine cases
    if not surya_pages and not chandra_pages:
        return []
    if not surya_pages:
        return chandra_pages
    if not chandra_pages:
        return surya_pages

    winners = []
    page_count = max(len(surya_pages), len(chandra_pages))

    for i in range(page_count):
        s_page = surya_pages[i] if i < len(surya_pages) else None
        c_page = chandra_pages[i] if i < len(chandra_pages) else None

        if s_page is None:
            winners.append(c_page)
            continue
        if c_page is None:
            winners.append(s_page)
            continue

        # Score both using the quality scorer
        s_score = _scorer.score(s_page.text)
        c_score = _scorer.score(c_page.text)

        # Update confidence with quality score
        s_page = PageResult(
            page_number=s_page.page_number, text=s_page.text,
            engine=s_page.engine, confidence=s_score,
            layout_html=s_page.layout_html,
        )
        c_page = PageResult(
            page_number=c_page.page_number, text=c_page.text,
            engine=c_page.engine, confidence=c_score,
            layout_html=c_page.layout_html,
        )

        winners.append(s_page if s_score >= c_score else c_page)

    return winners


def _assemble_document(
    winners: list[PageResult],
    source_path: Path,
    fmt: str,
    threshold: float = 0.6,
) -> LoadedDocument:
    """Assemble a LoadedDocument from per-page winners."""
    if not winners:
        return LoadedDocument(
            text="", source_path=source_path, format=fmt,
            page_count=0, quality_warning=True,
        )

    full_text = "\n\n".join(p.text for p in winners if p.text)
    engines_used = set(p.engine for p in winners if p.text)
    avg_confidence = (
        sum(p.confidence for p in winners) / len(winners) if winners else 0.0
    )
    any_below = any(p.confidence < threshold for p in winners)

    if len(engines_used) > 1:
        engine_label = "mixed"
    elif engines_used:
        engine_label = engines_used.pop()
    else:
        engine_label = None

    return LoadedDocument(
        text=full_text,
        source_path=source_path,
        format=fmt,
        page_count=len(winners),
        ocr_engine_used=engine_label,
        ocr_confidence=round(avg_confidence, 3),
        quality_warning=any_below,
        pages=winners,
    )
```

**Step 4: Run all document tests**

Run: `uv run pytest tests/test_loader_orchestrator.py tests/test_document_models.py tests/test_documents.py -x -v`
Expected: All pass

**Step 5: Commit**

```bash
git add mosaicx/documents/loader.py tests/test_loader_orchestrator.py
git commit -m "feat: rewrite document loader with dual-engine OCR orchestration"
```

---

### Task 8: Update config with new OCR fields

**Files:**
- Modify: `mosaicx/config.py`
- Modify: `tests/test_config.py`

**Step 1: Update config.py**

Add/update these fields in MosaicxConfig:

```python
    # --- Document loading ---
    ocr_engine: Literal["both", "surya", "chandra"] = "both"
    chandra_backend: Literal["vllm", "hf", "auto"] = "auto"
    chandra_server_url: str = ""
    quality_threshold: float = 0.6
    ocr_page_timeout: int = 60
    force_ocr: bool = False
    ocr_langs: list[str] = Field(default_factory=lambda: ["en", "de"])
```

Remove `vlm_model: str = "gemma3:27b"` (replaced by chandra_backend).

Add `Literal` imports to include the new literals.

**Step 2: Update tests**

Add to `tests/test_config.py`:

```python
    def test_ocr_config_defaults(self):
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig()
        assert cfg.ocr_engine == "both"
        assert cfg.chandra_backend == "auto"
        assert cfg.quality_threshold == 0.6
        assert cfg.ocr_page_timeout == 60
        assert cfg.force_ocr is False
        assert cfg.ocr_langs == ["en", "de"]

    def test_ocr_engine_env_override(self, monkeypatch):
        from mosaicx.config import MosaicxConfig

        monkeypatch.setenv("MOSAICX_OCR_ENGINE", "surya")
        cfg = MosaicxConfig()
        assert cfg.ocr_engine == "surya"
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_config.py -x -v`
Expected: All pass

**Step 4: Commit**

```bash
git add mosaicx/config.py tests/test_config.py
git commit -m "feat: add OCR engine config (ocr_engine, chandra_backend, quality_threshold)"
```

---

### Task 9: Wire config to loader and CLI

**Files:**
- Modify: `mosaicx/cli.py` — pass OCR config to load_document calls
- Modify: `mosaicx/__init__.py` — pass OCR config in public API wrappers

**Step 1: Update CLI extract command**

In the `extract` command body, change `load_document(path)` calls to pass OCR config:

```python
cfg = get_config()
doc = load_document(
    document,
    ocr_engine=cfg.ocr_engine,
    force_ocr=cfg.force_ocr,
    ocr_langs=cfg.ocr_langs,
    chandra_backend=cfg.chandra_backend if cfg.chandra_backend != "auto" else None,
    quality_threshold=cfg.quality_threshold,
    page_timeout=cfg.ocr_page_timeout,
)
```

Apply the same pattern to all CLI commands that call `load_document()`: `extract`, `batch`, `summarize`, `deidentify`.

**Step 2: Update public API in __init__.py**

Same pattern — the `extract()` and `summarize()` functions should pass OCR config to `load_document()`.

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -x -v --tb=short`
Expected: All pass

**Step 4: Commit**

```bash
git add mosaicx/cli.py mosaicx/__init__.py
git commit -m "feat: wire OCR config to loader in CLI and public API"
```

---

### Task 10: Update existing tests and add integration tests

**Files:**
- Modify: `tests/test_documents.py` — update imports
- Modify: `tests/integration/test_e2e.py` — update document loading test
- Create: `tests/test_ocr_integration.py` — slow tests for real OCR

**Step 1: Fix existing test imports**

In `tests/test_documents.py`, update any imports of `LoadedDocument` from `mosaicx.documents.loader` to `mosaicx.documents.models` (though both should work via re-export).

Verify existing tests still pass.

**Step 2: Add OCR integration tests (marked slow)**

```python
# tests/test_ocr_integration.py
"""Integration tests for OCR engines — require GPU and installed models.

Run with: pytest tests/test_ocr_integration.py -m slow -v
"""

import pytest
from pathlib import Path
from PIL import Image


@pytest.mark.slow
class TestOCRIntegration:
    def test_surya_on_image(self):
        """Surya can OCR a simple text image."""
        from mosaicx.documents.engines.surya_engine import SuryaEngine

        # Create a simple image with text (real test would use a scanned doc)
        img = Image.new("RGB", (200, 50), "white")
        engine = SuryaEngine()
        results = engine.ocr_pages([img], langs=["en"])
        assert len(results) == 1
        assert results[0].engine == "surya"

    def test_chandra_on_image(self):
        """Chandra can OCR a simple image."""
        from mosaicx.documents.engines.chandra_engine import ChandraEngine

        img = Image.new("RGB", (200, 50), "white")
        engine = ChandraEngine()
        results = engine.ocr_pages([img])
        assert len(results) == 1
        assert results[0].engine == "chandra"

    def test_full_pipeline_on_image(self, tmp_path):
        """Full load_document pipeline on an image file."""
        from mosaicx.documents.loader import load_document

        img = Image.new("RGB", (200, 100), "white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        doc = load_document(img_path)
        assert doc.source_path == img_path
        assert doc.format == "png"
```

**Step 3: Run tests**

Run: `uv run pytest tests/ -x -v --tb=short -k "not slow"`
Expected: All non-slow tests pass

**Step 4: Commit**

```bash
git add tests/test_documents.py tests/integration/test_e2e.py tests/test_ocr_integration.py
git commit -m "test: update document tests for dual-engine OCR, add integration tests"
```

---

## Summary

| Task | Deliverable | Tests |
|------|------------|-------|
| 1 | Replace docling with surya+chandra+pypdfium2 in pyproject.toml | N/A |
| 2 | Data models: PageResult, LoadedDocument (extended), DocumentLoadError | 8 tests |
| 3 | Quality scorer with medical vocabulary (~200 terms) | 12 tests |
| 4 | OCR engine protocol + pdf_to_images + image_to_pages | 5 tests |
| 5 | Surya engine wrapper | 3 tests |
| 6 | Chandra engine wrapper | 4 tests |
| 7 | Loader orchestrator rewrite (parallel dispatch + quality scoring) | 7 tests |
| 8 | Config: ocr_engine, chandra_backend, quality_threshold, etc. | 2 tests |
| 9 | Wire config to CLI + public API | Existing tests |
| 10 | Update tests + OCR integration tests | 3 slow tests |

**Total: 10 tasks, ~44 new tests**

Each task follows RED-GREEN-REFACTOR: write failing test -> implement -> verify pass -> commit.

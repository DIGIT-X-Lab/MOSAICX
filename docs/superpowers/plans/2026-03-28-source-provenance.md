# Source Provenance & Coordinate Mapping — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable GUI highlighting by mapping pipeline outputs (extracted fields, redacted PHI) back to source document locations (page number, bounding box coordinates).

**Architecture:** Three layers — (1) Document layer captures TextBlock coordinates during text extraction from pypdfium2/PaddleOCR, (2) Pipeline layer runs a post-extraction GatherEvidence LLM step to produce field-to-excerpt mappings, (3) Resolution layer matches excerpts to source text positions and looks up coordinates. Provenance is opt-in via `--provenance` flag.

**Tech Stack:** Python 3.11+, pypdfium2 (charbox API), PaddleOCR (dt_polys), DSPy (GatherEvidence signature), difflib (fuzzy matching)

**Spec:** `docs/superpowers/specs/2026-03-28-source-provenance-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `mosaicx/documents/models.py` | TextBlock dataclass, LoadedDocument extensions | Modify |
| `mosaicx/documents/loader.py` | Build TextBlocks during PDF text extraction | Modify |
| `mosaicx/documents/engines/paddleocr_engine.py` | Extract dt_polys from PaddleOCR results | Modify |
| `mosaicx/pipelines/provenance.py` | GatherEvidence, resolve_provenance, locate_in_document, fuzzy matching | Create |
| `mosaicx/pipelines/deidentifier.py` | Enrich redaction map with coordinates | Modify |
| `mosaicx/pipelines/extraction.py` | Call evidence step after forward() | Modify |
| `mosaicx/pipelines/radiology.py` | Call evidence step after forward() | Modify |
| `mosaicx/pipelines/pathology.py` | Call evidence step after forward() | Modify |
| `mosaicx/pipelines/summarizer.py` | Call evidence step after forward() | Modify |
| `mosaicx/sdk.py` | Preserve LoadedDocument, provenance param, _build_document_meta extension | Modify |
| `mosaicx/envelope.py` | Add provenance_requested flag | Modify |
| `mosaicx/cli.py` | --provenance flag, output threading | Modify |
| `mosaicx/mcp_server.py` | provenance param on tools | Modify |
| `tests/test_provenance.py` | Unit tests for provenance resolution + locate | Create |
| `tests/test_textblock_builders.py` | Unit tests for pypdfium2/PaddleOCR TextBlock construction | Create |
| `tests/test_provenance_integration.py` | Integration tests for end-to-end provenance | Create |

---

## Chunk 1: Document Layer — Models + TextBlock Lookup

### Task 1: TextBlock Model + LoadedDocument Extensions

**Files:**
- Modify: `mosaicx/documents/models.py:15-46`
- Test: `tests/test_provenance.py` (new)

- [ ] **Step 1: Write failing tests for TextBlock and LoadedDocument**

Create `tests/test_provenance.py`:

```python
"""Tests for source provenance: TextBlock, locate_in_document, resolve_provenance."""
from __future__ import annotations

import pytest


class TestTextBlock:
    """TextBlock dataclass construction and fields."""

    def test_create_text_block(self):
        from mosaicx.documents.models import TextBlock

        tb = TextBlock(
            text="Hello",
            start=0,
            end=5,
            page=1,
            bbox=(10.0, 20.0, 50.0, 35.0),
        )
        assert tb.text == "Hello"
        assert tb.start == 0
        assert tb.end == 5
        assert tb.page == 1
        assert tb.bbox == (10.0, 20.0, 50.0, 35.0)

    def test_text_block_fields_are_required(self):
        from mosaicx.documents.models import TextBlock

        with pytest.raises(TypeError):
            TextBlock(text="Hello")  # missing required fields


class TestLoadedDocumentExtensions:
    """LoadedDocument now carries text_blocks and page_dimensions."""

    def test_loaded_document_has_text_blocks(self):
        from pathlib import Path
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(text="hello", source_path=Path("/tmp/test.txt"), format="txt")
        assert doc.text_blocks == []

    def test_loaded_document_has_page_dimensions(self):
        from pathlib import Path
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(text="hello", source_path=Path("/tmp/test.txt"), format="txt")
        assert doc.page_dimensions == []

    def test_loaded_document_with_text_blocks(self):
        from pathlib import Path
        from mosaicx.documents.models import LoadedDocument, TextBlock

        blocks = [
            TextBlock(text="Hello", start=0, end=5, page=1, bbox=(10, 20, 50, 35)),
            TextBlock(text="world", start=6, end=11, page=1, bbox=(55, 20, 95, 35)),
        ]
        doc = LoadedDocument(
            text="Hello world",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            text_blocks=blocks,
            page_dimensions=[(612.0, 792.0)],
        )
        assert len(doc.text_blocks) == 2
        assert doc.page_dimensions == [(612.0, 792.0)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_provenance.py::TestTextBlock -v`
Expected: FAIL — `TextBlock` does not exist

- [ ] **Step 3: Implement TextBlock and LoadedDocument extensions**

In `mosaicx/documents/models.py`, add after the existing imports (line 8):

```python
@dataclass
class TextBlock:
    """A contiguous text region with its location on the source page.

    Internal data structure for coordinate mapping. Not serialized to
    output JSON — only resolved provenance records appear in output.
    """

    text: str
    start: int    # char offset in full document text
    end: int      # char offset in full document text
    page: int     # 1-indexed page number
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in page points
```

Add two new fields to `LoadedDocument` (after `pages` field, before the properties):

```python
    text_blocks: list[TextBlock] = field(default_factory=list)
    page_dimensions: list[tuple[float, float]] = field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/documents/models.py tests/test_provenance.py
git commit -m "feat(provenance): add TextBlock model and LoadedDocument extensions"
```

---

### Task 2: locate_in_document() Function

**Files:**
- Create: `mosaicx/pipelines/provenance.py`
- Test: `tests/test_provenance.py` (extend)

- [ ] **Step 1: Write failing tests for locate_in_document**

Append to `tests/test_provenance.py`:

```python
class TestLocateInDocument:
    """locate_in_document: map char ranges to page+bbox coordinates."""

    def _make_doc(self, text, blocks, page_dims=None):
        from pathlib import Path
        from mosaicx.documents.models import LoadedDocument

        return LoadedDocument(
            text=text,
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            text_blocks=blocks,
            page_dimensions=page_dims or [(612.0, 792.0)],
        )

    def test_single_block_exact_match(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import locate_in_document

        blocks = [TextBlock("Hello", 0, 5, 1, (10, 20, 50, 35))]
        doc = self._make_doc("Hello", blocks)
        result = locate_in_document(doc, 0, 5)
        assert result is not None
        assert len(result) == 1
        assert result[0]["page"] == 1
        assert result[0]["bbox"] == (10, 20, 50, 35)

    def test_range_spanning_multiple_blocks_same_page(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import locate_in_document

        blocks = [
            TextBlock("Hello", 0, 5, 1, (10, 20, 50, 35)),
            TextBlock(" ", 5, 6, 1, (50, 20, 55, 35)),
            TextBlock("world", 6, 11, 1, (55, 20, 95, 35)),
        ]
        doc = self._make_doc("Hello world", blocks)
        result = locate_in_document(doc, 0, 11)
        assert result is not None
        assert len(result) == 1  # same page -> merged
        assert result[0]["page"] == 1
        # Union bbox: min x0, min y0, max x1, max y1
        assert result[0]["bbox"] == (10, 20, 95, 35)

    def test_range_spanning_multiple_pages(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import locate_in_document

        blocks = [
            TextBlock("Page1", 0, 5, 1, (10, 20, 50, 35)),
            TextBlock("Page2", 7, 12, 2, (10, 700, 50, 715)),
        ]
        doc = self._make_doc("Page1\n\nPage2", blocks, [(612, 792), (612, 792)])
        result = locate_in_document(doc, 0, 12)
        assert result is not None
        assert len(result) == 2
        assert result[0]["page"] == 1
        assert result[1]["page"] == 2

    def test_no_text_blocks_returns_none(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = self._make_doc("Hello", [])
        result = locate_in_document(doc, 0, 5)
        assert result is None

    def test_range_outside_blocks_returns_none(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import locate_in_document

        blocks = [TextBlock("Hello", 0, 5, 1, (10, 20, 50, 35))]
        doc = self._make_doc("Hello world", blocks)
        result = locate_in_document(doc, 6, 11)
        assert result is None

    def test_partial_overlap_with_blocks(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import locate_in_document

        blocks = [
            TextBlock("Hello", 0, 5, 1, (10, 20, 50, 35)),
            TextBlock("world", 6, 11, 1, (55, 20, 95, 35)),
        ]
        doc = self._make_doc("Hello world", blocks)
        # Range overlaps only second block
        result = locate_in_document(doc, 6, 11)
        assert result is not None
        assert len(result) == 1
        assert result[0]["bbox"] == (55, 20, 95, 35)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_provenance.py::TestLocateInDocument -v`
Expected: FAIL — module `mosaicx.pipelines.provenance` does not exist

- [ ] **Step 3: Implement locate_in_document**

Create `mosaicx/pipelines/provenance.py`:

```python
"""Source provenance: map pipeline outputs to source document coordinates.

Provides:
- ``locate_in_document`` — character range → page + bounding box lookup
- ``resolve_provenance`` — evidence excerpts → source coordinates
- ``GatherEvidence`` — DSPy signature for post-extraction evidence gathering
"""
from __future__ import annotations

import bisect
from typing import Any

from mosaicx.documents.models import LoadedDocument


def locate_in_document(
    doc: LoadedDocument,
    start: int,
    end: int,
) -> list[dict[str, Any]] | None:
    """Find page + bbox for a character range in the source document.

    Uses binary search over ``doc.text_blocks``. If the range spans
    multiple pages, returns one entry per page with a union bbox for
    that page's overlapping blocks.

    Returns ``None`` if no text blocks cover the range (e.g., plain
    text documents with no coordinate data).
    """
    if not doc.text_blocks:
        return None

    # Find blocks that overlap [start, end)
    # text_blocks are sorted by .start
    block_starts = [b.start for b in doc.text_blocks]
    lo = bisect.bisect_right(block_starts, start) - 1
    lo = max(lo, 0)

    overlapping: list = []
    for i in range(lo, len(doc.text_blocks)):
        b = doc.text_blocks[i]
        if b.start >= end:
            break
        # Block overlaps [start, end) if b.start < end and b.end > start
        if b.end > start:
            overlapping.append(b)

    if not overlapping:
        return None

    # Group by page and compute union bbox per page
    pages: dict[int, list] = {}
    for b in overlapping:
        pages.setdefault(b.page, []).append(b)

    result = []
    for page_num in sorted(pages):
        blocks = pages[page_num]
        x0 = min(b.bbox[0] for b in blocks)
        y0 = min(b.bbox[1] for b in blocks)
        x1 = max(b.bbox[2] for b in blocks)
        y1 = max(b.bbox[3] for b in blocks)
        page_start = max(start, min(b.start for b in blocks))
        page_end = min(end, max(b.end for b in blocks))
        result.append({
            "page": page_num,
            "bbox": (x0, y0, x1, y1),
            "start": page_start,
            "end": page_end,
        })

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/pipelines/provenance.py tests/test_provenance.py
git commit -m "feat(provenance): add locate_in_document with binary search over TextBlocks"
```

---

## Chunk 2: TextBlock Builders — pypdfium2 + PaddleOCR

### Task 3: pypdfium2 TextBlock Builder

**Files:**
- Modify: `mosaicx/documents/loader.py:107-142`
- Test: `tests/test_textblock_builders.py` (new)

- [ ] **Step 1: Write failing tests for pypdfium2 builder**

Create `tests/test_textblock_builders.py`:

```python
"""Tests for TextBlock construction from pypdfium2 and PaddleOCR backends."""
from __future__ import annotations

import pytest
from pathlib import Path

SAMPLE_PDF = Path("tests/datasets/extract/sample_patient_vitals.pdf")


class TestPypdfium2TextBlocks:
    """TextBlock construction from native PDF text layer."""

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_loads_text_blocks_from_pdf(self):
        from mosaicx.documents.loader import load_document

        doc = load_document(SAMPLE_PDF)
        assert len(doc.text_blocks) > 0, "Expected TextBlocks from native PDF"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_blocks_cover_full_text(self):
        from mosaicx.documents.loader import load_document

        doc = load_document(SAMPLE_PDF)
        # Every non-whitespace character should be covered by a block
        covered = set()
        for tb in doc.text_blocks:
            covered.update(range(tb.start, tb.end))
        for i, ch in enumerate(doc.text):
            if not ch.isspace():
                assert i in covered, f"Char {i} ({ch!r}) not covered by any TextBlock"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_blocks_sorted_by_start(self):
        from mosaicx.documents.loader import load_document

        doc = load_document(SAMPLE_PDF)
        starts = [tb.start for tb in doc.text_blocks]
        assert starts == sorted(starts), "TextBlocks must be sorted by start offset"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_blocks_have_valid_bboxes(self):
        from mosaicx.documents.loader import load_document

        doc = load_document(SAMPLE_PDF)
        for tb in doc.text_blocks:
            x0, y0, x1, y1 = tb.bbox
            assert x1 >= x0, f"Invalid bbox x: {tb.bbox}"
            assert y1 >= y0, f"Invalid bbox y: {tb.bbox}"

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_page_dimensions_captured(self):
        from mosaicx.documents.loader import load_document

        doc = load_document(SAMPLE_PDF)
        assert len(doc.page_dimensions) > 0
        w, h = doc.page_dimensions[0]
        assert w > 0 and h > 0

    @pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
    def test_text_block_text_matches_source(self):
        """Critical invariant: TextBlock.text == doc.text[start:end]."""
        from mosaicx.documents.loader import load_document

        doc = load_document(SAMPLE_PDF)
        for tb in doc.text_blocks:
            actual = doc.text[tb.start:tb.end]
            assert actual == tb.text, (
                f"TextBlock text mismatch at [{tb.start}:{tb.end}]: "
                f"{tb.text!r} != {actual!r}"
            )

    def test_plain_text_has_no_text_blocks(self):
        from mosaicx.documents.loader import load_document

        txt_path = Path("tests/datasets/extract/sample_patient_note.txt")
        if not txt_path.exists():
            pytest.skip("sample text file missing")
        doc = load_document(txt_path)
        assert doc.text_blocks == []
        assert doc.page_dimensions == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_textblock_builders.py::TestPypdfium2TextBlocks -v`
Expected: FAIL — `text_blocks` is empty (not built yet)

- [ ] **Step 3: Implement pypdfium2 TextBlock builder**

In `mosaicx/documents/loader.py`, replace `_try_native_pdf_text()` (lines 107-142) with:

```python
def _try_native_pdf_text(path: Path) -> Optional[LoadedDocument]:
    """Try to extract text from a PDF's native text layer.

    Returns a LoadedDocument if the PDF has a usable text layer (>50 chars),
    or None if OCR is needed.  When successful, also builds word-level
    TextBlocks with bounding boxes for provenance mapping.
    """
    try:
        import pypdfium2
    except ImportError:
        return None

    try:
        pdf = pypdfium2.PdfDocument(str(path))
        pages_text: list[str] = []
        all_blocks: list = []
        page_dims: list[tuple[float, float]] = []
        global_offset = 0

        for page_idx, page in enumerate(pdf):
            tp = page.get_textpage()
            text = tp.get_text_bounded()
            pages_text.append(text)
            page_dims.append((float(page.get_width()), float(page.get_height())))

            # Build word-level TextBlocks from character boxes
            page_blocks = _build_textblocks_from_textpage(
                tp, text, page_num=page_idx + 1, global_offset=global_offset,
            )
            all_blocks.extend(page_blocks)

            # Advance offset: text length + "\n\n" join separator
            global_offset += len(text) + 2  # +2 for "\n\n"

            tp.close()
            page.close()
        pdf.close()

        full_text = "\n\n".join(pages_text)
        # If native text is too sparse, fall back to OCR
        if len(full_text.strip()) < 50:
            return None

        return LoadedDocument(
            text=full_text,
            source_path=path,
            format="pdf",
            page_count=len(pages_text),
            text_blocks=all_blocks,
            page_dimensions=page_dims,
        )
    except Exception:
        logger.debug("Native PDF text extraction failed for %s", path)
        return None


def _build_textblocks_from_textpage(
    textpage,
    page_text: str,
    page_num: int,
    global_offset: int,
) -> list:
    """Build word-level TextBlocks from a pypdfium2 textpage.

    Groups adjacent characters into words (split on whitespace).
    Returns a list of TextBlock with bbox coordinates in page points.
    """
    from mosaicx.documents.models import TextBlock

    blocks: list[TextBlock] = []
    n_chars = len(page_text)
    if n_chars == 0:
        return blocks

    word_start: int | None = None
    word_chars: list[tuple[int, tuple[float, float, float, float]]] = []

    for i in range(n_chars):
        ch = page_text[i]
        if ch.isspace():
            if word_chars:
                _flush_word(blocks, word_chars, word_start, page_text,
                            page_num, global_offset)
                word_chars = []
                word_start = None
        else:
            if word_start is None:
                word_start = i
            try:
                box = textpage.get_charbox(i)
                # pypdfium2 returns (left, bottom, right, top) in PDF coords
                # Normalize to (x0, y0, x1, y1) where y0 < y1
                x0, y_bottom, x1, y_top = box
                word_chars.append((i, (x0, min(y_bottom, y_top),
                                       x1, max(y_bottom, y_top))))
            except Exception:
                # Some chars may not have boxes (e.g., control chars)
                pass

    # Flush last word
    if word_chars:
        _flush_word(blocks, word_chars, word_start, page_text,
                    page_num, global_offset)

    return blocks


def _flush_word(
    blocks: list,
    word_chars: list[tuple[int, tuple[float, float, float, float]]],
    word_start: int | None,
    page_text: str,
    page_num: int,
    global_offset: int,
) -> None:
    """Flush accumulated word characters into a TextBlock."""
    from mosaicx.documents.models import TextBlock

    if not word_chars or word_start is None:
        return

    word_end = word_chars[-1][0] + 1
    text = page_text[word_start:word_end]

    x0 = min(c[1][0] for c in word_chars)
    y0 = min(c[1][1] for c in word_chars)
    x1 = max(c[1][2] for c in word_chars)
    y1 = max(c[1][3] for c in word_chars)

    blocks.append(TextBlock(
        text=text,
        start=global_offset + word_start,
        end=global_offset + word_end,
        page=page_num,
        bbox=(x0, y0, x1, y1),
    ))
```

Also add the import at the top of `loader.py` — `TextBlock` is imported inside the helper functions (lazy) so no top-level import needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_textblock_builders.py::TestPypdfium2TextBlocks -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run existing document tests for regression**

Run: `.venv/bin/python -m pytest tests/test_documents.py tests/test_document_models.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add mosaicx/documents/loader.py tests/test_textblock_builders.py
git commit -m "feat(provenance): build word-level TextBlocks from pypdfium2 charboxes"
```

---

### Task 4: PaddleOCR TextBlock Builder

**Files:**
- Modify: `mosaicx/documents/engines/paddleocr_engine.py:94-116`
- Modify: `mosaicx/documents/loader.py:177-211` (`_assemble_document`)
- Test: `tests/test_textblock_builders.py` (extend)

- [ ] **Step 1: Write failing tests for PaddleOCR builder**

Append to `tests/test_textblock_builders.py`:

```python
class TestPaddleOCRTextBlocks:
    """TextBlock construction from PaddleOCR polygon coordinates."""

    def test_polygon_to_bbox_conversion(self):
        """dt_polys quadrilateral -> axis-aligned (x0, y0, x1, y1)."""
        from mosaicx.documents.engines.paddleocr_engine import _polygon_to_bbox

        # 4-point polygon: [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
        poly = [[10, 20], [100, 20], [100, 40], [10, 40]]
        bbox = _polygon_to_bbox(poly)
        assert bbox == (10, 20, 100, 40)

    def test_polygon_to_bbox_rotated(self):
        """Rotated quadrilateral gives axis-aligned bounding box."""
        from mosaicx.documents.engines.paddleocr_engine import _polygon_to_bbox

        poly = [[15, 10], [105, 25], [100, 45], [10, 30]]
        bbox = _polygon_to_bbox(poly)
        assert bbox == (10, 10, 105, 45)

    def test_build_textblocks_from_ocr_result(self):
        """Simulate OCR result with rec_texts + dt_polys."""
        from mosaicx.documents.engines.paddleocr_engine import _build_ocr_textblocks

        rec_texts = ["Hello", "world"]
        dt_polys = [
            [[10, 20], [50, 20], [50, 35], [10, 35]],
            [[55, 20], [95, 20], [95, 35], [55, 35]],
        ]
        blocks = _build_ocr_textblocks(rec_texts, dt_polys, page_num=1, global_offset=0)
        assert len(blocks) == 2
        assert blocks[0].text == "Hello"
        assert blocks[0].start == 0
        assert blocks[0].end == 5
        assert blocks[0].bbox == (10, 20, 50, 35)
        assert blocks[1].text == "world"
        assert blocks[1].start == 6  # "Hello\n" -> offset 6
        assert blocks[1].end == 11

    def test_build_textblocks_empty_input(self):
        from mosaicx.documents.engines.paddleocr_engine import _build_ocr_textblocks

        blocks = _build_ocr_textblocks([], [], page_num=1, global_offset=0)
        assert blocks == []

    def test_build_textblocks_mismatched_lengths(self):
        """If dt_polys shorter than rec_texts, skip blocks without polys."""
        from mosaicx.documents.engines.paddleocr_engine import _build_ocr_textblocks

        rec_texts = ["Hello", "world", "extra"]
        dt_polys = [
            [[10, 20], [50, 20], [50, 35], [10, 35]],
            [[55, 20], [95, 20], [95, 35], [55, 35]],
        ]
        blocks = _build_ocr_textblocks(rec_texts, dt_polys, page_num=1, global_offset=0)
        # Should build blocks for the 2 that have polys
        assert len(blocks) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_textblock_builders.py::TestPaddleOCRTextBlocks -v`
Expected: FAIL — `_polygon_to_bbox` and `_build_ocr_textblocks` do not exist

- [ ] **Step 3: Implement PaddleOCR helpers**

In `mosaicx/documents/engines/paddleocr_engine.py`, add after the imports (line 18):

```python
def _polygon_to_bbox(
    poly: list[list[float]],
) -> tuple[float, float, float, float]:
    """Convert a 4-point polygon to an axis-aligned bounding box.

    Parameters
    ----------
    poly : list of [x, y] points (4 corners of a quadrilateral)

    Returns
    -------
    (x0, y0, x1, y1) — axis-aligned bounding box
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _build_ocr_textblocks(
    rec_texts: list[str],
    dt_polys: list[list[list[float]]],
    page_num: int,
    global_offset: int,
) -> list:
    """Build TextBlocks from PaddleOCR rec_texts + dt_polys.

    One TextBlock per text block. Character offsets computed from
    cumulative ``"\\n".join(rec_texts)``.
    """
    from mosaicx.documents.models import TextBlock

    blocks: list[TextBlock] = []
    offset = global_offset
    n = min(len(rec_texts), len(dt_polys))

    for i in range(len(rec_texts)):
        text = rec_texts[i]
        if i < n:
            bbox = _polygon_to_bbox(dt_polys[i])
            blocks.append(TextBlock(
                text=text,
                start=offset,
                end=offset + len(text),
                page=page_num,
                bbox=bbox,
            ))
        # Advance offset: text length + 1 for "\n" join separator
        offset += len(text) + 1  # +1 for "\n"

    return blocks
```

- [ ] **Step 4: Update PaddleOCR engine to extract dt_polys and build TextBlocks**

In `paddleocr_engine.py`, modify the `ocr_pages` method. After line 98 (`rec_scores = res.get("rec_scores", [])`), add:

```python
                    dt_polys = res.get("dt_polys", [])
```

Then modify `PageResult` creation (lines 110-117) to store blocks in metadata:

```python
                page_results.append(
                    PageResult(
                        page_number=i + 1,
                        text=text,
                        engine="paddleocr",
                        confidence=round(confidence, 4),
                        layout_html=None,
                        # Store TextBlock data for provenance
                        _text_blocks=_build_ocr_textblocks(
                            rec_texts, dt_polys, page_num=i + 1, global_offset=0,
                        ),
                    )
                )
```

Wait — `PageResult` doesn't have a `_text_blocks` field. Instead, store in the metadata approach. Actually, the simpler approach: add an optional field to `PageResult`.

In `mosaicx/documents/models.py`, add to `PageResult`:

```python
    text_blocks: list = field(default_factory=list)  # TextBlock list for provenance
```

Then in `paddleocr_engine.py`, pass the blocks:

```python
                    ocr_blocks = _build_ocr_textblocks(
                        rec_texts, dt_polys if dt_polys else [],
                        page_num=i + 1, global_offset=0,
                    )
```

And in the `PageResult` construction, add `text_blocks=ocr_blocks`.

- [ ] **Step 5: Update _assemble_document to merge TextBlocks**

In `mosaicx/documents/loader.py`, modify `_assemble_document()` (lines 177-211). After `full_text = "\n\n".join(...)` (line 190), add:

```python
    # Merge TextBlocks from per-page results, adjusting global offsets
    all_blocks: list = []
    global_offset = 0
    for p in winners:
        if p.text:
            for tb in p.text_blocks:
                from mosaicx.documents.models import TextBlock
                all_blocks.append(TextBlock(
                    text=tb.text,
                    start=tb.start + global_offset,
                    end=tb.end + global_offset,
                    page=tb.page,
                    bbox=tb.bbox,
                ))
            global_offset += len(p.text) + 2  # +2 for "\n\n"
```

Then add `text_blocks=all_blocks` to the `LoadedDocument` constructor at the end.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_textblock_builders.py -v`
Expected: All tests PASS

- [ ] **Step 7: Run regression tests**

Run: `.venv/bin/python -m pytest tests/test_documents.py tests/test_document_models.py tests/test_provenance.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add mosaicx/documents/models.py mosaicx/documents/engines/paddleocr_engine.py mosaicx/documents/loader.py tests/test_textblock_builders.py
git commit -m "feat(provenance): build TextBlocks from PaddleOCR dt_polys + merge in assembly"
```

---

## Chunk 3: Provenance Resolution — Fuzzy Matching + GatherEvidence

### Task 5: resolve_provenance() with Exact + Fuzzy Matching

**Files:**
- Modify: `mosaicx/pipelines/provenance.py`
- Test: `tests/test_provenance.py` (extend)

- [ ] **Step 1: Write failing tests for resolve_provenance**

Append to `tests/test_provenance.py`:

```python
class TestResolveProvenance:
    """resolve_provenance: map evidence excerpts to source coordinates."""

    def _make_doc_with_blocks(self, text, blocks, page_dims=None):
        from pathlib import Path
        from mosaicx.documents.models import LoadedDocument

        return LoadedDocument(
            text=text,
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            text_blocks=blocks,
            page_dimensions=page_dims or [(612.0, 792.0)],
        )

    def test_exact_match(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import resolve_provenance

        text = "The ejection fraction is 55% and wall motion is normal."
        blocks = [TextBlock(text, 0, len(text), 1, (10, 20, 400, 35))]
        doc = self._make_doc_with_blocks(text, blocks)

        evidence = {"ef": "ejection fraction is 55%"}
        result = resolve_provenance(doc, evidence)

        assert "ef" in result
        assert result["ef"]["resolution"] == "exact"
        assert result["ef"]["start"] == 4
        assert result["ef"]["end"] == 26

    def test_fuzzy_match(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import resolve_provenance

        text = "The left ventricular ejection fraction was estimated at approximately 55 percent."
        blocks = [TextBlock(text, 0, len(text), 1, (10, 20, 500, 35))]
        doc = self._make_doc_with_blocks(text, blocks)

        # LLM slightly paraphrased
        evidence = {"ef": "left ventricular ejection fraction was estimated at approximately 55 percent"}
        result = resolve_provenance(doc, evidence)

        assert "ef" in result
        assert result["ef"]["resolution"] in ("exact", "fuzzy")
        assert result["ef"]["start"] >= 0

    def test_unresolved_when_no_match(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import resolve_provenance

        text = "Normal chest X-ray."
        blocks = [TextBlock(text, 0, len(text), 1, (10, 20, 200, 35))]
        doc = self._make_doc_with_blocks(text, blocks)

        evidence = {"finding": "This text is completely different and not in the document at all xyz"}
        result = resolve_provenance(doc, evidence)

        assert result["finding"]["resolution"] == "unresolved"

    def test_multiple_fields(self):
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import resolve_provenance

        text = "EF is 55%. Wall motion normal. No pericardial effusion."
        blocks = [TextBlock(text, 0, len(text), 1, (10, 20, 500, 35))]
        doc = self._make_doc_with_blocks(text, blocks)

        evidence = {
            "ef": "EF is 55%",
            "wall_motion": "Wall motion normal",
            "pericardium": "No pericardial effusion",
        }
        result = resolve_provenance(doc, evidence)

        assert len(result) == 3
        for key in evidence:
            assert key in result
            assert result[key]["resolution"] in ("exact", "fuzzy")

    def test_no_text_blocks_still_returns_offsets(self):
        """Plain text docs: no spans but start/end populated."""
        from mosaicx.pipelines.provenance import resolve_provenance
        from pathlib import Path
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(
            text="The EF is 55%.",
            source_path=Path("/tmp/note.txt"),
            format="txt",
        )
        evidence = {"ef": "EF is 55%"}
        result = resolve_provenance(doc, evidence)

        assert result["ef"]["resolution"] == "exact"
        assert result["ef"]["start"] == 4
        assert result["ef"]["end"] == 13
        assert result["ef"]["spans"] == []  # no blocks -> no spans

    def test_short_excerpt_higher_threshold(self):
        """Short excerpts (<40 chars) need 0.90 similarity to avoid false positives."""
        from mosaicx.documents.models import TextBlock
        from mosaicx.pipelines.provenance import resolve_provenance

        # Repetitive text with similar short phrases
        text = "No evidence of fracture. No evidence of dislocation. No evidence of effusion."
        blocks = [TextBlock(text, 0, len(text), 1, (10, 20, 500, 35))]
        doc = self._make_doc_with_blocks(text, blocks)

        evidence = {"finding": "No evidence of fracture"}
        result = resolve_provenance(doc, evidence)

        assert result["finding"]["resolution"] == "exact"
        # Should match the FIRST occurrence
        assert result["finding"]["start"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_provenance.py::TestResolveProvenance -v`
Expected: FAIL — `resolve_provenance` not yet defined

- [ ] **Step 3: Implement resolve_provenance**

Add to `mosaicx/pipelines/provenance.py`:

```python
import difflib


def resolve_provenance(
    doc: LoadedDocument,
    evidence: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Resolve evidence excerpts to source coordinates.

    For each field, finds the excerpt in ``doc.text`` and looks up
    page/bbox coordinates from the document's TextBlock map.

    Resolution tiers:
        1. Exact match — ``str.find()``
        2. Fuzzy match — ``difflib.SequenceMatcher`` (threshold varies by length)
        3. Unresolved — no match found

    Parameters
    ----------
    doc : LoadedDocument
        Source document with text and optional TextBlocks.
    evidence : dict[str, str]
        Mapping of field names to verbatim excerpts from the source.

    Returns
    -------
    dict[str, dict]
        Mapping of field names to provenance records.
    """
    claimed: set[tuple[int, int]] = set()  # track claimed positions
    results: dict[str, dict[str, Any]] = {}

    for field_name, excerpt in evidence.items():
        record = _resolve_single(doc, excerpt, claimed)
        results[field_name] = record

    return results


def _resolve_single(
    doc: LoadedDocument,
    excerpt: str,
    claimed: set[tuple[int, int]],
) -> dict[str, Any]:
    """Resolve a single excerpt to source coordinates."""
    text = doc.text

    # Tier 1: Exact match
    start = 0
    while True:
        pos = text.find(excerpt, start)
        if pos == -1:
            break
        end = pos + len(excerpt)
        if (pos, end) not in claimed:
            claimed.add((pos, end))
            spans = locate_in_document(doc, pos, end) or []
            return {
                "excerpt": excerpt,
                "start": pos,
                "end": end,
                "spans": [{"page": s["page"], "bbox": s["bbox"]} for s in spans],
                "resolution": "exact",
            }
        start = pos + 1

    # Tier 2: Fuzzy match
    threshold = 0.90 if len(excerpt) < 40 else 0.80
    best = _fuzzy_find(text, excerpt, threshold)
    if best is not None:
        pos, end, ratio = best
        if (pos, end) not in claimed:
            claimed.add((pos, end))
            spans = locate_in_document(doc, pos, end) or []
            return {
                "excerpt": excerpt,
                "start": pos,
                "end": end,
                "spans": [{"page": s["page"], "bbox": s["bbox"]} for s in spans],
                "resolution": "fuzzy",
            }

    # Tier 3: Unresolved
    return {
        "excerpt": excerpt,
        "start": -1,
        "end": -1,
        "spans": [],
        "resolution": "unresolved",
    }


def _fuzzy_find(
    text: str,
    excerpt: str,
    threshold: float,
) -> tuple[int, int, float] | None:
    """Find the best fuzzy match for excerpt in text.

    Slides a window of len(excerpt) +/- 20% over the text and picks
    the position with the highest SequenceMatcher ratio above threshold.

    Returns (start, end, ratio) or None.
    """
    excerpt_len = len(excerpt)
    if excerpt_len == 0:
        return None

    min_window = max(1, int(excerpt_len * 0.8))
    max_window = int(excerpt_len * 1.2)

    best_ratio = 0.0
    best_pos = -1
    best_end = -1

    # Use a stride for efficiency on long texts
    stride = max(1, excerpt_len // 4)

    for window_size in range(min_window, max_window + 1, max(1, (max_window - min_window) // 3)):
        for i in range(0, len(text) - window_size + 1, stride):
            candidate = text[i:i + window_size]
            ratio = difflib.SequenceMatcher(None, excerpt, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i
                best_end = i + window_size

    if best_ratio >= threshold:
        # Refine: search around best_pos with stride=1
        refine_start = max(0, best_pos - stride)
        refine_end_pos = min(len(text), best_pos + stride + excerpt_len)
        for window_size in range(min_window, max_window + 1):
            for i in range(refine_start, min(refine_end_pos, len(text) - window_size + 1)):
                candidate = text[i:i + window_size]
                ratio = difflib.SequenceMatcher(None, excerpt, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i
                    best_end = i + window_size

        return (best_pos, best_end, best_ratio)

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/pipelines/provenance.py tests/test_provenance.py
git commit -m "feat(provenance): resolve_provenance with exact/fuzzy/unresolved tiers"
```

---

### Task 6: GatherEvidence DSPy Signature

**Files:**
- Modify: `mosaicx/pipelines/provenance.py`
- Test: `tests/test_provenance.py` (extend)

- [ ] **Step 1: Write failing test for GatherEvidence signature**

Append to `tests/test_provenance.py`:

```python
class TestGatherEvidence:
    """GatherEvidence DSPy signature exists and has correct fields."""

    def test_signature_has_required_fields(self):
        from mosaicx.pipelines.provenance import GatherEvidence

        fields = GatherEvidence.signature.fields if hasattr(GatherEvidence, 'signature') else {}
        # Check input/output field names exist on the class
        assert hasattr(GatherEvidence, '__signature__') or 'document_text' in str(dir(GatherEvidence))

    def test_gather_evidence_importable(self):
        from mosaicx.pipelines.provenance import GatherEvidence
        assert GatherEvidence is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_provenance.py::TestGatherEvidence -v`
Expected: FAIL — `GatherEvidence` not defined

- [ ] **Step 3: Implement GatherEvidence using lazy loading pattern**

Add to `mosaicx/pipelines/provenance.py`, following the project's lazy loading pattern:

```python
def _build_dspy_classes():
    """Lazily build DSPy classes for provenance evidence gathering."""
    import dspy

    class GatherEvidence(dspy.Signature):
        """Given a source document and extracted fields, find the verbatim
        excerpt from the source document that supports each field value.

        For each field, copy the exact substring from document_text that
        contains or directly states the field's value. Do not paraphrase.
        """

        document_text: str = dspy.InputField(
            desc="Full text of the source document"
        )
        extracted_fields: str = dspy.InputField(
            desc="JSON object of field names and their extracted values"
        )
        evidence: str = dspy.OutputField(
            desc="JSON object mapping each field name to the verbatim excerpt "
                 "from document_text that supports its value. Each excerpt must "
                 "be an exact substring of document_text."
        )

    return {"GatherEvidence": GatherEvidence}


_provenance_dspy_classes: dict | None = None
_PROVENANCE_CLASS_NAMES = frozenset({"GatherEvidence"})


def __getattr__(name: str):
    """Module-level __getattr__ for lazy loading of DSPy classes."""
    global _provenance_dspy_classes

    if name in _PROVENANCE_CLASS_NAMES:
        if _provenance_dspy_classes is None:
            _provenance_dspy_classes = _build_dspy_classes()
        return _provenance_dspy_classes[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Also add a convenience function for running evidence gathering:

```python
def gather_evidence(
    document_text: str,
    extracted_fields: dict[str, Any],
) -> dict[str, str]:
    """Run the GatherEvidence LLM step and parse the result.

    Parameters
    ----------
    document_text : str
        Full source document text.
    extracted_fields : dict
        Field name → extracted value mapping.

    Returns
    -------
    dict[str, str]
        Field name → verbatim excerpt from source.
    """
    import json
    import dspy

    evidence_step = dspy.Predict(GatherEvidence)
    result = evidence_step(
        document_text=document_text,
        extracted_fields=json.dumps(extracted_fields, default=str, ensure_ascii=False),
    )

    try:
        evidence = json.loads(result.evidence)
    except (json.JSONDecodeError, TypeError):
        evidence = {}

    return evidence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/pipelines/provenance.py tests/test_provenance.py
git commit -m "feat(provenance): GatherEvidence DSPy signature with lazy loading"
```

---

## Chunk 4: Pipeline Integration — Deidentifier + Extraction

### Task 7: Deidentifier Coordinate Enrichment

**Files:**
- Modify: `mosaicx/pipelines/deidentifier.py:480-517`
- Test: `tests/test_provenance.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/test_provenance.py`:

```python
class TestDeidentifierProvenance:
    """Deidentifier redaction map enriched with coordinates."""

    def test_enrich_redaction_map_with_coordinates(self):
        from mosaicx.pipelines.provenance import enrich_redaction_map
        from mosaicx.documents.models import LoadedDocument, TextBlock
        from pathlib import Path

        text = "Patient Name: Jane Doe\nPatient ID: PID-123"
        blocks = [
            TextBlock("Patient", 0, 7, 1, (10, 20, 60, 35)),
            TextBlock("Name:", 8, 13, 1, (65, 20, 100, 35)),
            TextBlock("Jane", 14, 18, 1, (105, 20, 135, 35)),
            TextBlock("Doe", 19, 22, 1, (140, 20, 165, 35)),
        ]
        doc = LoadedDocument(
            text=text, source_path=Path("/tmp/t.pdf"), format="pdf",
            text_blocks=blocks, page_dimensions=[(612, 792)],
        )
        redaction_map = [
            {"original": "Jane Doe", "replacement": "[REDACTED]",
             "start": 14, "end": 22, "phi_type": "NAME", "method": "llm"},
        ]

        enriched = enrich_redaction_map(doc, redaction_map)
        assert len(enriched) == 1
        entry = enriched[0]
        assert "spans" in entry
        assert len(entry["spans"]) == 1
        assert entry["spans"][0]["page"] == 1
        assert entry["resolution"] == "exact"
        assert "excerpt" in entry

    def test_enrich_without_text_blocks(self):
        """Plain text: spans empty, but start/end still present."""
        from mosaicx.pipelines.provenance import enrich_redaction_map
        from mosaicx.documents.models import LoadedDocument
        from pathlib import Path

        doc = LoadedDocument(
            text="Patient Name: Jane Doe",
            source_path=Path("/tmp/t.txt"), format="txt",
        )
        redaction_map = [
            {"original": "Jane Doe", "replacement": "[REDACTED]",
             "start": 14, "end": 22, "phi_type": "NAME", "method": "llm"},
        ]

        enriched = enrich_redaction_map(doc, redaction_map)
        assert enriched[0]["spans"] == []
        assert enriched[0]["start"] == 14
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_provenance.py::TestDeidentifierProvenance -v`
Expected: FAIL — `enrich_redaction_map` not defined

- [ ] **Step 3: Implement enrich_redaction_map**

Add to `mosaicx/pipelines/provenance.py`:

```python
def enrich_redaction_map(
    doc: LoadedDocument,
    redaction_map: list[dict[str, Any]],
    context_chars: int = 30,
) -> list[dict[str, Any]]:
    """Enrich redaction map entries with page coordinates and excerpts.

    Parameters
    ----------
    doc : LoadedDocument
        Source document with optional TextBlocks.
    redaction_map : list[dict]
        Redaction entries with ``start`` and ``end`` character offsets.
    context_chars : int
        Number of surrounding characters to include in excerpt.

    Returns
    -------
    list[dict]
        Enriched redaction map with ``spans``, ``excerpt``, ``resolution``.
    """
    enriched = []
    for entry in redaction_map:
        new_entry = dict(entry)
        start = entry["start"]
        end = entry["end"]

        # Build excerpt with surrounding context
        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(doc.text), end + context_chars)
        new_entry["excerpt"] = doc.text[ctx_start:ctx_end]

        # Look up coordinates
        spans_raw = locate_in_document(doc, start, end)
        if spans_raw:
            new_entry["spans"] = [{"page": s["page"], "bbox": s["bbox"]} for s in spans_raw]
            new_entry["resolution"] = "exact"
        else:
            new_entry["spans"] = []
            new_entry["resolution"] = "exact"  # offsets are exact, just no coords

        enriched.append(new_entry)
    return enriched
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_provenance.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add mosaicx/pipelines/provenance.py tests/test_provenance.py
git commit -m "feat(provenance): enrich_redaction_map with page coordinates"
```

---

### Task 8: SDK — Preserve LoadedDocument + provenance Parameter

**Files:**
- Modify: `mosaicx/sdk.py:278-375` (`_resolve_documents`), `251-275` (`_build_document_meta`)
- Modify: `mosaicx/envelope.py:39-92`
- Test: `tests/test_sdk_envelope.py` (extend)

- [ ] **Step 1: Write failing test**

Append to `tests/test_provenance.py`:

```python
class TestSDKDocumentPreservation:
    """SDK preserves LoadedDocument for provenance."""

    def test_resolve_documents_returns_loaded_document(self):
        from pathlib import Path
        from mosaicx.sdk import _resolve_documents

        path = Path("tests/datasets/extract/sample_patient_note.txt")
        if not path.exists():
            pytest.skip("sample file missing")

        results = _resolve_documents(path)
        assert len(results) == 1
        # Fourth element should be LoadedDocument
        assert len(results[0]) == 4
        _, _, _, loaded_doc = results[0]
        from mosaicx.documents.models import LoadedDocument
        assert isinstance(loaded_doc, LoadedDocument)

    def test_build_document_meta_includes_page_dimensions(self):
        from pathlib import Path
        from mosaicx.sdk import _build_document_meta
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(
            text="test", source_path=Path("/tmp/t.pdf"), format="pdf",
            page_count=1, page_dimensions=[(612.0, 792.0)],
        )
        meta = _build_document_meta(doc)
        assert "page_dimensions" in meta
        assert meta["page_dimensions"] == [(612.0, 792.0)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_provenance.py::TestSDKDocumentPreservation -v`
Expected: FAIL — tuple length is 3, not 4

- [ ] **Step 3: Modify _resolve_documents to preserve LoadedDocument**

In `mosaicx/sdk.py`, change `_resolve_documents()` (line 278) return type and logic:

The function currently returns `list[tuple[str, str, dict]]`. Change to return `list[tuple[str, str, dict, LoadedDocument]]` by keeping the `LoadedDocument` in the tuple.

Find every place that calls `_resolve_documents()` and update the unpacking — most places do:
```python
for filepath, text, meta in results:
```
Change to:
```python
for filepath, text, meta, loaded_doc in results:
```

In `_build_document_meta()`, add `page_dimensions`:
```python
    meta["page_dimensions"] = list(doc.page_dimensions) if doc.page_dimensions else []
```

- [ ] **Step 4: Update envelope.py — rename provenance to provenance_requested**

In `mosaicx/envelope.py`, change the `provenance` parameter name to `provenance_requested` in `build_envelope()` and in the returned dict key.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_provenance.py::TestSDKDocumentPreservation tests/test_sdk_envelope.py -v`
Expected: PASS

- [ ] **Step 6: Run full regression**

Run: `.venv/bin/python -m pytest tests/ --ignore=tests/datasets -x -q`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add mosaicx/sdk.py mosaicx/envelope.py tests/test_provenance.py
git commit -m "feat(provenance): preserve LoadedDocument in SDK, add page_dimensions to meta"
```

---

## Chunk 5: Pipeline Evidence Integration + CLI/MCP Threading

### Task 9: Wire Provenance into Deidentifier CLI + SDK Path

**Files:**
- Modify: `mosaicx/cli.py:2263-2401` (deidentify command)
- Modify: `mosaicx/sdk.py:1445-1473` (`_deidentify_single_text`)
- Modify: `mosaicx/pipelines/deidentifier.py:435-517`

- [ ] **Step 1: Add --provenance flag to deidentify CLI command**

In `mosaicx/cli.py`, add a `--provenance` option to the `deidentify` command (after existing options):

```python
@click.option("--provenance", is_flag=True, default=False,
              help="Include source provenance (page coordinates) in output.")
```

Thread `provenance` and the `LoadedDocument` through to `enrich_redaction_map()` when the flag is set.

- [ ] **Step 2: Update SDK deidentify path**

In `mosaicx/sdk.py`, add `provenance: bool = False` parameter to `deidentify()` and `_deidentify_single_text()`. When True:
- Pass `LoadedDocument` to the deidentifier
- Call `enrich_redaction_map()` on the result
- Include `page_dimensions` in the `_mosaicx._document` envelope

- [ ] **Step 3: Run deidentify tests**

Run: `.venv/bin/python -m pytest tests/test_deidentifier_pipeline.py tests/test_provenance.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add mosaicx/cli.py mosaicx/sdk.py mosaicx/pipelines/deidentifier.py
git commit -m "feat(provenance): wire provenance into deidentify CLI + SDK"
```

---

### Task 10: Wire Provenance into Extract CLI + SDK Path

**Files:**
- Modify: `mosaicx/cli.py:570-687` (extract command)
- Modify: `mosaicx/sdk.py:1046-1120` (`_extract_single_text`)
- Modify: `mosaicx/pipelines/extraction.py`

- [ ] **Step 1: Add --provenance to extract command (already may exist — check first)**

The `extract` CLI command may already have a `--provenance` flag (check line ~620). If not, add it.

When `--provenance` is set:
1. After pipeline `forward()` returns, call `gather_evidence(doc.text, extracted_fields)`
2. Call `resolve_provenance(doc, evidence)`
3. Include `provenance` dict in output

- [ ] **Step 2: Update SDK extract path**

In `_extract_single_text()`, when `provenance=True`:
1. After extraction completes, call `gather_evidence()`
2. Call `resolve_provenance()`
3. Add `"provenance"` key to output dict

- [ ] **Step 3: Run extract tests**

Run: `.venv/bin/python -m pytest tests/test_extraction_pipeline.py tests/test_cli_extract.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add mosaicx/cli.py mosaicx/sdk.py mosaicx/pipelines/extraction.py
git commit -m "feat(provenance): wire provenance into extract CLI + SDK"
```

---

### Task 11: Wire Provenance into Radiology + Pathology + Summarizer

**Files:**
- Modify: `mosaicx/pipelines/radiology.py:249-258` (after forward return)
- Modify: `mosaicx/pipelines/pathology.py:148-158` (after forward return)
- Modify: `mosaicx/pipelines/summarizer.py:170-173` (after forward return)

- [ ] **Step 1: Add provenance wrapper to each pipeline**

For each pipeline, add a wrapper function (outside the DSPy Module) that:
1. Calls `forward()` as usual
2. If provenance requested, runs `gather_evidence()` + `resolve_provenance()`
3. Returns prediction + provenance dict

This keeps the DSPy Module's `forward()` untouched and adds evidence as a post-processing step.

- [ ] **Step 2: Wire through CLI and SDK for each mode**

Update the mode-dispatch logic in CLI `extract` command and SDK to call the provenance wrapper when the flag is set.

- [ ] **Step 3: Run pipeline tests**

Run: `.venv/bin/python -m pytest tests/test_radiology_pipeline.py tests/test_pathology_pipeline.py tests/test_summarizer_pipeline.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add mosaicx/pipelines/radiology.py mosaicx/pipelines/pathology.py mosaicx/pipelines/summarizer.py mosaicx/cli.py mosaicx/sdk.py
git commit -m "feat(provenance): wire provenance into radiology, pathology, summarizer pipelines"
```

---

### Task 12: MCP Server Provenance Parameter

**Files:**
- Modify: `mosaicx/mcp_server.py:159-327` (extract_document), `331-368` (deidentify_text)

- [ ] **Step 1: Add provenance parameter to MCP tools**

Add `provenance: bool = False` parameter to both `extract_document()` and `deidentify_text()` tools. When True, include provenance in the JSON response.

- [ ] **Step 2: Run MCP tests**

Run: `.venv/bin/python -m pytest tests/test_mcp_server_dspy_config.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add mosaicx/mcp_server.py
git commit -m "feat(provenance): add provenance parameter to MCP server tools"
```

---

## Chunk 6: Integration Testing

### Task 13: End-to-End Integration Test

**Files:**
- Create: `tests/test_provenance_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""Integration tests for source provenance end-to-end."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

SAMPLE_PDF = Path("tests/datasets/extract/sample_patient_vitals.pdf")
SAMPLE_TXT = Path("tests/datasets/extract/sample_patient_note.txt")


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
class TestDeidentifyProvenance:
    """Deidentify with provenance on a real PDF."""

    def test_deidentify_provenance_output_structure(self):
        """Provenance output has correct structure for GUI consumption."""
        from mosaicx.documents.loader import load_document
        from mosaicx.pipelines.provenance import enrich_redaction_map

        doc = load_document(SAMPLE_PDF)
        assert len(doc.text_blocks) > 0, "PDF should have TextBlocks"

        # Simulate a redaction map (from the deidentifier)
        # Find "Sarah Johnson" in the text
        name = "Sarah Johnson"
        pos = doc.text.find(name)
        assert pos >= 0, f"{name!r} not found in PDF text"

        redaction_map = [{
            "original": name,
            "replacement": "[REDACTED]",
            "start": pos,
            "end": pos + len(name),
            "phi_type": "NAME",
            "method": "llm",
        }]

        enriched = enrich_redaction_map(doc, redaction_map)

        assert len(enriched) == 1
        entry = enriched[0]
        assert entry["spans"], "Should have page coordinates for PDF"
        assert entry["spans"][0]["page"] == 1
        bbox = entry["spans"][0]["bbox"]
        assert all(isinstance(v, (int, float)) for v in bbox)
        assert entry["resolution"] == "exact"

    def test_text_block_invariant(self):
        """TextBlock.text == doc.text[start:end] for all blocks."""
        from mosaicx.documents.loader import load_document

        doc = load_document(SAMPLE_PDF)
        for tb in doc.text_blocks:
            assert doc.text[tb.start:tb.end] == tb.text, (
                f"Invariant violation at [{tb.start}:{tb.end}]"
            )
```

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/python -m pytest tests/test_provenance_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ --ignore=tests/datasets -x -q`
Expected: All tests PASS, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_provenance_integration.py
git commit -m "test(provenance): add end-to-end integration tests"
```

---

### Task 14: Final Cleanup + Full Regression

- [ ] **Step 1: Run linter**

Run: `.venv/bin/python -m ruff check mosaicx/documents/models.py mosaicx/documents/loader.py mosaicx/documents/engines/paddleocr_engine.py mosaicx/pipelines/provenance.py --fix`

- [ ] **Step 2: Run type checker**

Run: `.venv/bin/python -m mypy mosaicx/pipelines/provenance.py mosaicx/documents/models.py`

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ --ignore=tests/datasets -q`
Expected: All tests PASS

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -u && git commit -m "style: lint fixes for provenance feature"
```

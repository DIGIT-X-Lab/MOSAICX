"""Tests for TextBlock model, LoadedDocument extensions, and locate_in_document."""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Task 1: TextBlock and LoadedDocument extensions
# ---------------------------------------------------------------------------


class TestTextBlock:
    def test_create_text_block(self):
        from mosaicx.documents.models import TextBlock

        block = TextBlock(
            text="Patient has bilateral opacity.",
            start=0,
            end=29,
            page=1,
            bbox=(10.0, 20.0, 200.0, 35.0),
        )
        assert block.text == "Patient has bilateral opacity."
        assert block.start == 0
        assert block.end == 29
        assert block.page == 1
        assert block.bbox == (10.0, 20.0, 200.0, 35.0)

    def test_text_block_fields_are_required(self):
        from mosaicx.documents.models import TextBlock

        # All five fields are required — omitting any should raise TypeError
        with pytest.raises(TypeError):
            TextBlock(text="hello", start=0, end=5, page=1)  # missing bbox

        with pytest.raises(TypeError):
            TextBlock(text="hello", start=0, end=5, bbox=(0, 0, 1, 1))  # missing page

        with pytest.raises(TypeError):
            TextBlock(text="hello", start=0, page=1, bbox=(0, 0, 1, 1))  # missing end

        with pytest.raises(TypeError):
            TextBlock(text="hello", end=5, page=1, bbox=(0, 0, 1, 1))  # missing start

        with pytest.raises(TypeError):
            TextBlock(start=0, end=5, page=1, bbox=(0, 0, 1, 1))  # missing text


class TestLoadedDocumentExtensions:
    def test_loaded_document_has_text_blocks(self):
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(
            text="Hello world",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
        )
        assert hasattr(doc, "text_blocks")
        assert doc.text_blocks == []

    def test_loaded_document_has_page_dimensions(self):
        from mosaicx.documents.models import LoadedDocument

        doc = LoadedDocument(
            text="Hello world",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
        )
        assert hasattr(doc, "page_dimensions")
        assert doc.page_dimensions == []

    def test_loaded_document_with_text_blocks(self):
        from mosaicx.documents.models import LoadedDocument, TextBlock

        blocks = [
            TextBlock(text="First line.", start=0, end=11, page=1, bbox=(10.0, 10.0, 100.0, 25.0)),
            TextBlock(text="Second line.", start=12, end=24, page=1, bbox=(10.0, 30.0, 100.0, 45.0)),
        ]
        doc = LoadedDocument(
            text="First line.\nSecond line.",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            text_blocks=blocks,
            page_dimensions=[(595.0, 842.0)],
        )
        assert len(doc.text_blocks) == 2
        assert doc.text_blocks[0].text == "First line."
        assert doc.text_blocks[1].start == 12
        assert doc.page_dimensions == [(595.0, 842.0)]

    def test_page_result_has_text_blocks(self):
        from mosaicx.documents.models import PageResult, TextBlock

        block = TextBlock(
            text="Liver unremarkable.",
            start=0,
            end=19,
            page=1,
            bbox=(10.0, 50.0, 200.0, 65.0),
        )
        page = PageResult(
            page_number=1,
            text="Liver unremarkable.",
            engine="surya",
            confidence=0.92,
            text_blocks=[block],
        )
        assert len(page.text_blocks) == 1
        assert page.text_blocks[0].text == "Liver unremarkable."

    def test_page_result_text_blocks_default_empty(self):
        from mosaicx.documents.models import PageResult

        page = PageResult(
            page_number=1,
            text="Text",
            engine="surya",
            confidence=0.9,
        )
        assert page.text_blocks == []


# ---------------------------------------------------------------------------
# Task 2: locate_in_document()
# ---------------------------------------------------------------------------


def _make_doc(blocks: list[dict]) -> "LoadedDocument":
    """Helper: build a LoadedDocument with TextBlock list from dicts."""
    from mosaicx.documents.models import LoadedDocument, TextBlock

    text_blocks = [
        TextBlock(
            text=b["text"],
            start=b["start"],
            end=b["end"],
            page=b["page"],
            bbox=tuple(b["bbox"]),
        )
        for b in blocks
    ]
    full_text = " ".join(b["text"] for b in blocks)
    return LoadedDocument(
        text=full_text,
        source_path=Path("/tmp/doc.pdf"),
        format="pdf",
        text_blocks=text_blocks,
    )


class TestLocateInDocument:
    def test_single_block_exact_match(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "Pleural effusion noted.", "start": 0, "end": 23, "page": 1, "bbox": [10, 20, 200, 35]},
        ])
        result = locate_in_document(doc, 0, 23)
        assert result is not None
        assert len(result) == 1
        assert result[0]["page"] == 1
        assert result[0]["bbox"] == (10, 20, 200, 35)
        assert result[0]["start"] == 0
        assert result[0]["end"] == 23

    def test_range_spanning_multiple_blocks_same_page(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "First block.", "start": 0, "end": 12, "page": 1, "bbox": [10, 10, 100, 25]},
            {"text": "Second block.", "start": 13, "end": 26, "page": 1, "bbox": [10, 30, 110, 45]},
            {"text": "Third block.", "start": 27, "end": 39, "page": 1, "bbox": [10, 50, 100, 65]},
        ])
        # Range covers first two blocks
        result = locate_in_document(doc, 0, 26)
        assert result is not None
        assert len(result) == 1  # same page => one entry
        entry = result[0]
        assert entry["page"] == 1
        # Union bbox: x0=min(10,10)=10, y0=min(10,30)=10, x1=max(100,110)=110, y1=max(25,45)=45
        assert entry["bbox"] == (10, 10, 110, 45)
        assert entry["start"] == 0
        assert entry["end"] == 26

    def test_range_spanning_multiple_pages(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "Page one text.", "start": 0, "end": 14, "page": 1, "bbox": [10, 700, 200, 715]},
            {"text": "Page two text.", "start": 15, "end": 29, "page": 2, "bbox": [10, 10, 200, 25]},
        ])
        result = locate_in_document(doc, 0, 29)
        assert result is not None
        assert len(result) == 2
        pages = {r["page"]: r for r in result}
        assert 1 in pages
        assert 2 in pages
        assert pages[1]["bbox"] == (10, 700, 200, 715)
        assert pages[2]["bbox"] == (10, 10, 200, 25)

    def test_no_text_blocks_returns_none(self):
        from mosaicx.documents.models import LoadedDocument
        from mosaicx.pipelines.provenance import locate_in_document

        doc = LoadedDocument(
            text="Some text without blocks.",
            source_path=Path("/tmp/empty.pdf"),
            format="pdf",
        )
        assert doc.text_blocks == []
        result = locate_in_document(doc, 0, 10)
        assert result is None

    def test_range_outside_blocks_returns_none(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "Block text.", "start": 50, "end": 61, "page": 1, "bbox": [10, 10, 100, 25]},
        ])
        # Query range is entirely before the block
        result = locate_in_document(doc, 0, 20)
        assert result is None

    def test_partial_overlap_with_blocks(self):
        from mosaicx.pipelines.provenance import locate_in_document

        doc = _make_doc([
            {"text": "Alpha block.", "start": 0, "end": 12, "page": 1, "bbox": [10, 10, 100, 25]},
            {"text": "Beta block.", "start": 20, "end": 31, "page": 1, "bbox": [10, 30, 100, 45]},
            {"text": "Gamma block.", "start": 40, "end": 52, "page": 2, "bbox": [10, 10, 100, 25]},
        ])
        # Range [5, 25] overlaps Alpha (0-12) and Beta (20-31), not Gamma (40-52)
        result = locate_in_document(doc, 5, 25)
        assert result is not None
        assert len(result) == 1  # both overlapping blocks are on page 1
        assert result[0]["page"] == 1
        # Union of Alpha and Beta bboxes
        assert result[0]["bbox"] == (10, 10, 100, 45)
        assert result[0]["start"] == 5
        assert result[0]["end"] == 25


# ---------------------------------------------------------------------------
# Task 5: resolve_provenance()
# ---------------------------------------------------------------------------


def _make_doc_from_text(text: str, blocks: list[dict] | None = None) -> "LoadedDocument":
    """Build a LoadedDocument from a text string with optional TextBlock list."""
    from mosaicx.documents.models import LoadedDocument, TextBlock

    text_blocks = []
    if blocks:
        text_blocks = [
            TextBlock(
                text=b["text"],
                start=b["start"],
                end=b["end"],
                page=b["page"],
                bbox=tuple(b["bbox"]),
            )
            for b in blocks
        ]
    return LoadedDocument(
        text=text,
        source_path=Path("/tmp/doc.pdf"),
        format="pdf",
        text_blocks=text_blocks,
    )


class TestResolveProvenance:
    def test_exact_match(self):
        from mosaicx.pipelines.provenance import resolve_provenance

        text = "Patient has bilateral pleural effusion. No pneumothorax."
        doc = _make_doc_from_text(
            text,
            [
                {
                    "text": "Patient has bilateral pleural effusion.",
                    "start": 0,
                    "end": 38,
                    "page": 1,
                    "bbox": [10, 10, 200, 25],
                },
                {
                    "text": "No pneumothorax.",
                    "start": 39,
                    "end": 55,
                    "page": 1,
                    "bbox": [10, 30, 200, 45],
                },
            ],
        )
        evidence = {"finding": "bilateral pleural effusion"}
        result = resolve_provenance(doc, evidence)

        assert "finding" in result
        entry = result["finding"]
        assert entry["excerpt"] == "bilateral pleural effusion"
        assert entry["resolution"] == "exact"
        assert entry["start"] >= 0
        assert entry["end"] > entry["start"]
        assert isinstance(entry["spans"], list)

    def test_fuzzy_match(self):
        from mosaicx.pipelines.provenance import resolve_provenance

        # Introduce a minor typo — should still fuzzy-match
        text = "Patient has bilateral pleural effusion noted on imaging."
        doc = _make_doc_from_text(text)
        evidence = {"finding": "bilateral pleural efusion"}  # typo: efusion
        result = resolve_provenance(doc, evidence)

        assert "finding" in result
        entry = result["finding"]
        # Should resolve via fuzzy (typo won't exact-match)
        assert entry["resolution"] in ("exact", "fuzzy")
        assert entry["start"] >= 0

    def test_unresolved_when_no_match(self):
        from mosaicx.pipelines.provenance import resolve_provenance

        doc = _make_doc_from_text("The liver appears normal in size and echogenicity.")
        evidence = {"finding": "completely unrelated text xyz"}
        result = resolve_provenance(doc, evidence)

        entry = result["finding"]
        assert entry["resolution"] == "unresolved"
        assert entry["start"] == -1
        assert entry["end"] == -1
        assert entry["spans"] == []

    def test_multiple_fields(self):
        from mosaicx.pipelines.provenance import resolve_provenance

        text = "Spleen is enlarged. Liver appears normal."
        doc = _make_doc_from_text(text)
        evidence = {
            "spleen": "Spleen is enlarged",
            "liver": "Liver appears normal",
        }
        result = resolve_provenance(doc, evidence)

        assert "spleen" in result
        assert "liver" in result
        assert result["spleen"]["resolution"] == "exact"
        assert result["liver"]["resolution"] == "exact"
        # Offsets should not overlap
        assert result["spleen"]["end"] <= result["liver"]["start"]

    def test_no_text_blocks_still_returns_offsets(self):
        from mosaicx.pipelines.provenance import resolve_provenance

        text = "Bilateral atelectasis noted."
        doc = _make_doc_from_text(text)  # no blocks
        evidence = {"finding": "Bilateral atelectasis noted"}
        result = resolve_provenance(doc, evidence)

        entry = result["finding"]
        assert entry["resolution"] == "exact"
        assert entry["start"] == 0
        assert entry["end"] == len("Bilateral atelectasis noted")
        # No text_blocks => spans is empty list
        assert entry["spans"] == []

    def test_short_excerpt_higher_threshold(self):
        """Short excerpts (< 40 chars) require >= 0.90 similarity."""
        from mosaicx.pipelines.provenance import resolve_provenance

        text = "Normal sinus rhythm."
        doc = _make_doc_from_text(text)
        # Completely different text — well below 0.90 for a short excerpt
        evidence = {"rhythm": "xyz abc defgh"}
        result = resolve_provenance(doc, evidence)

        entry = result["rhythm"]
        assert entry["resolution"] == "unresolved"


# ---------------------------------------------------------------------------
# Task 6: GatherEvidence DSPy Signature
# ---------------------------------------------------------------------------


class TestGatherEvidence:
    def test_gather_evidence_importable(self):
        """GatherEvidence should be importable from provenance without dspy error."""
        # This just tests that lazy loading works and the class is accessible.
        from mosaicx.pipelines import provenance

        cls = provenance.GatherEvidence
        assert cls is not None
        # It should be a dspy.Signature subclass
        import dspy

        assert issubclass(cls, dspy.Signature)

    def test_gather_evidence_has_expected_fields(self):
        from mosaicx.pipelines import provenance

        cls = provenance.GatherEvidence
        # DSPy stores fields in model_fields for Signature classes
        fields = cls.model_fields
        assert "document_text" in fields
        assert "extracted_fields" in fields
        assert "evidence" in fields


# ---------------------------------------------------------------------------
# Task 7: enrich_redaction_map
# ---------------------------------------------------------------------------


class TestDeidentifierProvenance:
    def test_enrich_redaction_map_with_coordinates(self):
        from mosaicx.pipelines.provenance import enrich_redaction_map

        text = "Patient: John Doe. DOB: 01/02/1990. Diagnosis: normal."
        blocks = [
            {
                "text": "Patient: John Doe.",
                "start": 0,
                "end": 18,
                "page": 1,
                "bbox": [10, 10, 200, 25],
            },
            {
                "text": "DOB: 01/02/1990.",
                "start": 19,
                "end": 35,
                "page": 1,
                "bbox": [10, 30, 200, 45],
            },
            {
                "text": "Diagnosis: normal.",
                "start": 36,
                "end": 54,
                "page": 1,
                "bbox": [10, 50, 200, 65],
            },
        ]
        doc = _make_doc_from_text(text, blocks)
        redaction_map = [
            {
                "original": "John Doe",
                "replacement": "[REDACTED]",
                "start": 9,
                "end": 17,
                "phi_type": "OTHER",
                "method": "llm",
            },
            {
                "original": "01/02/1990",
                "replacement": "[REDACTED]",
                "start": 24,
                "end": 34,
                "phi_type": "DATE",
                "method": "regex",
            },
        ]
        enriched = enrich_redaction_map(doc, redaction_map)

        assert len(enriched) == 2

        # First entry: "John Doe" within block 0 (start=0, end=18)
        e0 = enriched[0]
        assert e0["original"] == "John Doe"
        assert e0["resolution"] == "located"
        assert isinstance(e0["spans"], list)
        assert len(e0["spans"]) >= 1
        assert e0["spans"][0]["page"] == 1
        assert "excerpt" in e0

        # Second entry: date
        e1 = enriched[1]
        assert e1["original"] == "01/02/1990"
        assert e1["resolution"] == "located"
        assert isinstance(e1["spans"], list)

    def test_enrich_without_text_blocks(self):
        from mosaicx.pipelines.provenance import enrich_redaction_map

        text = "Patient: Jane Smith. MRN: 123456."
        doc = _make_doc_from_text(text)  # no text_blocks
        redaction_map = [
            {
                "original": "Jane Smith",
                "replacement": "[REDACTED]",
                "start": 9,
                "end": 19,
                "phi_type": "OTHER",
                "method": "llm",
            },
        ]
        enriched = enrich_redaction_map(doc, redaction_map)

        assert len(enriched) == 1
        e = enriched[0]
        assert e["original"] == "Jane Smith"
        # Without text blocks locate_in_document returns None => spans is empty
        assert e["spans"] == []
        assert e["resolution"] == "located"
        assert "excerpt" in e

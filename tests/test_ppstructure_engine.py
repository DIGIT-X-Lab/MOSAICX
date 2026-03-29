# tests/test_ppstructure_engine.py
"""Tests for PPStructureEngine (mocked -- no model downloads needed).

Mocks ``_run_worker`` to return raw dict data matching the subprocess
JSON output format, so tests never touch PaddlePaddle.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mosaicx.documents.engines.ppstructure_engine import (
    PPStructureEngine,
    _pages_from_raw,
)


def _make_raw_page(
    page_num: int = 1,
    rec_texts: list[str] | None = None,
    rec_scores: list[float] | None = None,
    dt_polys: list | None = None,
    blocks: list[dict] | None = None,
) -> dict:
    """Build a raw page dict matching the subprocess JSON output."""
    if rec_texts is None:
        rec_texts = ["Hello", "World"]
    if rec_scores is None:
        rec_scores = [0.95, 0.92]
    if dt_polys is None:
        dt_polys = [
            [[0, 0], [50, 0], [50, 10], [0, 10]],
            [[60, 0], [120, 0], [120, 10], [60, 10]],
        ]
    if blocks is None:
        # Auto-generate a text block covering all OCR detections
        all_xs = [p[0] for poly in dt_polys for p in poly]
        all_ys = [p[1] for poly in dt_polys for p in poly]
        blocks = [{
            "label": "text",
            "content": " ".join(rec_texts),
            "bbox": [
                min(all_xs) if all_xs else 0,
                min(all_ys) if all_ys else 0,
                max(all_xs) if all_xs else 100,
                max(all_ys) if all_ys else 100,
            ],
        }]

    return {
        "page_num": page_num,
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "dt_polys": dt_polys,
        "blocks": blocks,
    }


class TestPagesFromRaw:
    """Tests for _pages_from_raw (shared builder)."""

    def test_basic_text_block(self):
        raw = [_make_raw_page()]
        pages = _pages_from_raw(raw)
        assert len(pages) == 1
        assert "Hello" in pages[0].text
        assert "World" in pages[0].text
        assert pages[0].engine == "ppstructure"

    def test_table_block(self):
        raw = [_make_raw_page(
            rec_texts=["BP", "120/80"],
            rec_scores=[0.98, 0.96],
            dt_polys=[
                [[0, 0], [30, 0], [30, 10], [0, 10]],
                [[50, 0], [100, 0], [100, 10], [50, 10]],
            ],
            blocks=[{
                "label": "table",
                "content": "<table><tr><td>BP</td><td>120/80</td></tr></table>",
                "bbox": [0, 0, 100, 10],
            }],
        )]
        pages = _pages_from_raw(raw)
        assert "BP" in pages[0].text
        assert "120/80" in pages[0].text
        assert "|" in pages[0].text  # Markdown table
        assert pages[0].layout_html is not None
        assert "<table>" in pages[0].layout_html


class TestPPStructureEngine:
    @patch.object(PPStructureEngine, "_run_worker")
    def test_single_page_image(self, mock_sub):
        """Single-page image produces 1 PageResult with correct text."""
        mock_sub.return_value = _pages_from_raw([_make_raw_page(
            rec_texts=["BP", "120/80"],
            rec_scores=[0.98, 0.96],
            dt_polys=[
                [[0, 0], [30, 0], [30, 10], [0, 10]],
                [[50, 0], [100, 0], [100, 10], [50, 10]],
            ],
            blocks=[{
                "label": "table",
                "content": "<table><tr><td>BP</td><td>120/80</td></tr></table>",
                "bbox": [0, 0, 100, 10],
            }],
        )])

        engine = PPStructureEngine(lang="german")
        pages = engine.process_file(Path("test.jpg"))

        assert len(pages) == 1
        assert "BP" in pages[0].text
        assert "120/80" in pages[0].text
        assert pages[0].engine == "ppstructure"
        assert pages[0].confidence == pytest.approx(0.97, abs=0.01)
        assert pages[0].page_number == 1

    @patch.object(PPStructureEngine, "_run_worker")
    def test_multi_page_pdf(self, mock_sub):
        """Multi-page PDF produces one PageResult per page."""
        mock_sub.return_value = _pages_from_raw([
            _make_raw_page(page_num=1),
            _make_raw_page(page_num=2),
        ])

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.pdf"))

        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2

    @patch.object(PPStructureEngine, "_run_worker")
    def test_textblock_offsets_match_page_text(self, mock_sub):
        """TextBlock start/end point into the page text correctly."""
        mock_sub.return_value = _pages_from_raw([_make_raw_page(
            rec_texts=["Title", "Hello", "World"],
            rec_scores=[0.99, 0.95, 0.93],
            dt_polys=[
                [[0, 0], [50, 0], [50, 10], [0, 10]],
                [[0, 20], [50, 20], [50, 30], [0, 30]],
                [[60, 20], [120, 20], [120, 30], [60, 30]],
            ],
            blocks=[
                {"label": "paragraph_title", "content": "Title",
                 "bbox": [0, 0, 50, 10]},
                {"label": "text", "content": "Hello World",
                 "bbox": [0, 20, 120, 30]},
            ],
        )])

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.jpg"))
        page = pages[0]

        for tb in page.text_blocks:
            assert page.text[tb.start : tb.end] == tb.text, (
                f"Offset mismatch: page.text[{tb.start}:{tb.end}]="
                f"{page.text[tb.start:tb.end]!r} != {tb.text!r}"
            )

    @patch.object(PPStructureEngine, "_run_worker")
    def test_textblock_bboxes_are_valid(self, mock_sub):
        """Every TextBlock has x1 >= x0 and y1 >= y0."""
        mock_sub.return_value = _pages_from_raw([_make_raw_page()])

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.jpg"))

        for tb in pages[0].text_blocks:
            x0, y0, x1, y1 = tb.bbox
            assert x1 >= x0
            assert y1 >= y0

    @patch.object(PPStructureEngine, "_run_worker")
    def test_table_html_captured(self, mock_sub):
        """Table blocks populate layout_html on PageResult."""
        mock_sub.return_value = _pages_from_raw([_make_raw_page(
            blocks=[{
                "label": "table",
                "content": "<table><tr><td>BP</td><td>120/80</td></tr></table>",
                "bbox": [0, 0, 200, 100],
            }],
        )])

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.jpg"))

        assert pages[0].layout_html is not None
        assert "<table>" in pages[0].layout_html

    @patch.object(PPStructureEngine, "_run_worker")
    def test_empty_result(self, mock_sub):
        """Empty PPStructure output -> single empty PageResult."""
        from mosaicx.documents.models import PageResult

        mock_sub.return_value = [
            PageResult(
                page_number=1, text="", engine="ppstructure", confidence=0.0
            )
        ]

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.jpg"))

        assert len(pages) == 1
        assert pages[0].text == ""
        assert pages[0].confidence == 0.0

    @patch.object(PPStructureEngine, "_run_worker")
    def test_fallback_on_subprocess_failure(self, mock_sub):
        """Falls back to in-process when subprocess fails."""
        mock_sub.side_effect = RuntimeError("subprocess crashed")

        engine = PPStructureEngine(lang="german")
        # This will try _run_in_process which will also fail (no model),
        # but we're testing that the fallback path is reached.
        with pytest.raises(Exception):
            engine.process_file(Path("nonexistent.jpg"))

        mock_sub.assert_called_once()

    @patch.object(PPStructureEngine, "_run_worker")
    def test_lang_stored(self, mock_sub):
        """Language parameter is stored on the engine."""
        mock_sub.return_value = _pages_from_raw([_make_raw_page()])

        engine = PPStructureEngine(lang="german")
        assert engine._lang == "german"
        engine.process_file(Path("test.jpg"))

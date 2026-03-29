# tests/test_ppstructure_engine.py
"""Tests for PPStructureEngine (mocked -- no model downloads needed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_page_result(
    page_index: int = 0,
    page_count: int = 1,
    markdown_text: str = "Hello World",
    rec_texts: list[str] | None = None,
    rec_scores: list[float] | None = None,
    dt_polys: list | None = None,
    parsing_res_list: list | None = None,
):
    """Build a mock LayoutParsingResultV2 dict matching PPStructureV3 output."""
    if rec_texts is None:
        rec_texts = ["Hello", "World"]
    if rec_scores is None:
        rec_scores = [0.95, 0.92]
    if dt_polys is None:
        dt_polys = [
            [[0, 0], [50, 0], [50, 10], [0, 10]],
            [[60, 0], [120, 0], [120, 10], [60, 10]],
        ]
    if parsing_res_list is None:
        parsing_res_list = []

    data = {
        "page_index": page_index,
        "page_count": page_count,
        "overall_ocr_res": {
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "dt_polys": dt_polys,
        },
        "parsing_res_list": parsing_res_list,
        "table_res_list": [],
    }

    result = MagicMock()
    result.__getitem__ = lambda self, key: data[key]
    result.get = lambda key, default=None: data.get(key, default)
    result.keys = lambda: list(data.keys())
    result.markdown = {"markdown_texts": markdown_text}
    return result


class TestPPStructureEngine:
    @patch("mosaicx.documents.engines.ppstructure_engine._get_ppstructure")
    def test_single_page_image(self, mock_get):
        """Single-page image produces 1 PageResult with markdown text."""
        from mosaicx.documents.engines.ppstructure_engine import PPStructureEngine

        mock_engine = MagicMock()
        mock_engine.predict.return_value = [
            _make_mock_page_result(
                markdown_text="| BP | 120/80 |",
                rec_texts=["BP", "120/80"],
                rec_scores=[0.98, 0.96],
                dt_polys=[
                    [[0, 0], [30, 0], [30, 10], [0, 10]],
                    [[50, 0], [100, 0], [100, 10], [50, 10]],
                ],
            )
        ]
        mock_get.return_value = mock_engine

        engine = PPStructureEngine(lang="german")
        pages = engine.process_file(Path("test.jpg"))

        assert len(pages) == 1
        assert pages[0].text == "| BP | 120/80 |"
        assert pages[0].engine == "ppstructure"
        assert pages[0].confidence == pytest.approx(0.97, abs=0.01)
        assert pages[0].page_number == 1
        assert len(pages[0].text_blocks) == 2
        assert pages[0].text_blocks[0].text == "BP"
        assert pages[0].text_blocks[1].text == "120/80"

    @patch("mosaicx.documents.engines.ppstructure_engine._get_ppstructure")
    def test_multi_page_pdf(self, mock_get):
        """Multi-page PDF produces one PageResult per page."""
        from mosaicx.documents.engines.ppstructure_engine import PPStructureEngine

        mock_engine = MagicMock()
        mock_engine.predict.return_value = [
            _make_mock_page_result(
                page_index=0, page_count=2, markdown_text="Page one"
            ),
            _make_mock_page_result(
                page_index=1, page_count=2, markdown_text="Page two"
            ),
        ]
        mock_get.return_value = mock_engine

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.pdf"))

        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[0].text == "Page one"
        assert pages[1].page_number == 2
        assert pages[1].text == "Page two"

    @patch("mosaicx.documents.engines.ppstructure_engine._get_ppstructure")
    def test_textblock_offsets_match_markdown(self, mock_get):
        """TextBlock start/end point into the markdown text correctly."""
        from mosaicx.documents.engines.ppstructure_engine import PPStructureEngine

        md = "## Title\n\nHello World"
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [
            _make_mock_page_result(
                markdown_text=md,
                rec_texts=["Title", "Hello", "World"],
                rec_scores=[0.99, 0.95, 0.93],
                dt_polys=[
                    [[0, 0], [50, 0], [50, 10], [0, 10]],
                    [[0, 20], [50, 20], [50, 30], [0, 30]],
                    [[60, 20], [120, 20], [120, 30], [60, 30]],
                ],
            )
        ]
        mock_get.return_value = mock_engine

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.jpg"))
        page = pages[0]

        for tb in page.text_blocks:
            assert page.text[tb.start : tb.end] == tb.text, (
                f"Offset mismatch: page.text[{tb.start}:{tb.end}]="
                f"{page.text[tb.start:tb.end]!r} != {tb.text!r}"
            )

    @patch("mosaicx.documents.engines.ppstructure_engine._get_ppstructure")
    def test_textblock_bboxes_are_valid(self, mock_get):
        """Every TextBlock has x1 >= x0 and y1 >= y0."""
        from mosaicx.documents.engines.ppstructure_engine import PPStructureEngine

        mock_engine = MagicMock()
        mock_engine.predict.return_value = [_make_mock_page_result()]
        mock_get.return_value = mock_engine

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.jpg"))

        for tb in pages[0].text_blocks:
            x0, y0, x1, y1 = tb.bbox
            assert x1 >= x0
            assert y1 >= y0

    @patch("mosaicx.documents.engines.ppstructure_engine._get_ppstructure")
    def test_table_html_captured(self, mock_get):
        """Table blocks populate layout_html on PageResult."""
        from mosaicx.documents.engines.ppstructure_engine import PPStructureEngine

        table_block = MagicMock()
        table_block.label = "table"
        table_block.content = (
            "<table><tr><td>BP</td><td>120/80</td></tr></table>"
        )
        table_block.bbox = [0, 0, 200, 100]

        mock_engine = MagicMock()
        mock_engine.predict.return_value = [
            _make_mock_page_result(parsing_res_list=[table_block])
        ]
        mock_get.return_value = mock_engine

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.jpg"))

        assert pages[0].layout_html is not None
        assert "<table>" in pages[0].layout_html

    @patch("mosaicx.documents.engines.ppstructure_engine._get_ppstructure")
    def test_empty_result(self, mock_get):
        """Empty PPStructure output -> single empty PageResult."""
        from mosaicx.documents.engines.ppstructure_engine import PPStructureEngine

        mock_engine = MagicMock()
        mock_engine.predict.return_value = []
        mock_get.return_value = mock_engine

        engine = PPStructureEngine()
        pages = engine.process_file(Path("test.jpg"))

        assert len(pages) == 1
        assert pages[0].text == ""
        assert pages[0].confidence == 0.0

    @patch("mosaicx.documents.engines.ppstructure_engine._get_ppstructure")
    def test_lang_passed_through(self, mock_get):
        """Language parameter is forwarded to PPStructureV3."""
        from mosaicx.documents.engines.ppstructure_engine import PPStructureEngine

        mock_engine = MagicMock()
        mock_engine.predict.return_value = [_make_mock_page_result()]
        mock_get.return_value = mock_engine

        engine = PPStructureEngine(lang="german")
        engine.process_file(Path("test.jpg"))

        mock_get.assert_called_once_with("german")

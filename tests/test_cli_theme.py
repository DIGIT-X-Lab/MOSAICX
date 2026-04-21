"""Tests for CLI theme banner rendering."""
from __future__ import annotations

from io import StringIO
from rich.console import Console


def test_print_banner_renders_model_and_ocr_lines():
    """Banner status strip shows model line and OCR line with badges."""
    from mosaicx.cli_theme import print_banner

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    print_banner(
        "2.0.0",
        console,
        lm="openai/gpt-oss:120b",
        num_ctx=131072,
        inference_engine="ollama 0.9.2",
        ocr_engine="paddleocr",
        ocr_langs=["en", "de"],
    )

    output = buf.getvalue()
    assert "gpt-oss:120b" in output
    assert "131k" in output
    assert "ollama 0.9.2" in output
    assert "paddleocr" in output
    assert "en, de" in output


def test_print_banner_omits_via_when_no_engine():
    """When inference_engine is empty, skip the 'via' segment."""
    from mosaicx.cli_theme import print_banner

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    print_banner(
        "2.0.0",
        console,
        lm="openai/gpt-oss:120b",
        num_ctx=131072,
        inference_engine="",
        ocr_engine="paddleocr",
    )

    output = buf.getvalue()
    assert "gpt-oss:120b" in output
    assert "via" not in output


def test_print_banner_minimal_lm_only():
    """With only lm set, model line renders and OCR line is skipped."""
    from mosaicx.cli_theme import print_banner

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    print_banner("2.0.0", console, lm="gemma3:27b")

    output = buf.getvalue()
    assert "gemma3:27b" in output

"""Command-line interface wiring for MOSAICX."""

from .app import cli, generate, list_schemas_cmd, extract, summarize, main

__all__ = [
    "cli",
    "generate",
    "list_schemas_cmd",
    "extract",
    "summarize",
    "main",
]

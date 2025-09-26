"""MOSAICX CLI entry point."""

from __future__ import annotations

from .cli import cli, extract, generate, list_schemas_cmd, main, summarize

__all__ = [
    "cli",
    "generate",
    "list_schemas_cmd",
    "extract",
    "summarize",
    "main",
]


if __name__ == "__main__":
    main()

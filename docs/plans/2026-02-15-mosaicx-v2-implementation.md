# MOSAICX v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite MOSAICX as a DSPy-powered medical document structuring platform with RadReport templates, ontology resolution, completeness scoring, FHIR export, and batch processing.

**Architecture:** Single DSPy backend replacing 3 LLM libraries. Composable pipeline modules (extraction, radiology, schema gen, summarizer, deidentifier). YAML template compiler for user-defined schemas. Hybrid ontology resolution (local lookup + LLM fallback). 3-layer completeness scoring. GEPA-optimizable with progressive fallback.

**Tech Stack:** Python 3.11+, DSPy >=2.6, Pydantic >=2.0, Click >=8.1, Rich >=13.0, Docling >=2.0, PyArrow, Pandas

**Design doc:** `docs/plans/2026-02-15-mosaicx-v2-rewrite-design.md`

**Working directory:** `/Users/nutellabear/Documents/00-Code/MOSAICX`

**Test command:** `uv run pytest tests/ -x -q` (project uses uv and .venv)

---

## Phase 1: Core Infrastructure

### Task 1: Update pyproject.toml dependencies and version

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update dependencies**

Replace the current dependencies block with the v2 deps. Remove: `openai`, `ollama`, `instructor`, `outlines`, `json-repair`, `httpx`, `requests`, `python-cfonts`, `rich-click`. Add: `dspy>=2.6`, `pydantic-settings>=2.0`, `pandas>=2.0`, `pyarrow>=14.0`. Keep: `click`, `rich`, `pydantic`, `docling`, `pyyaml`, `typing-extensions`.

```toml
dependencies = [
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "dspy>=2.6.0",
    "docling>=2.0.0",
    "pyyaml>=6.0",
    "typing-extensions>=4.8.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
]
```

Add optional dependency groups:

```toml
[project.optional-dependencies]
pdf = ["reportlab>=4.4"]
docx = ["python-docx>=1.0"]
hf = ["datasets>=2.0"]
all = ["mosaicx[pdf,docx,hf]"]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.6.0",
    "mypy>=1.10.0",
    "pre-commit>=3.3.0",
]
```

Bump version to `"2.0.0a1"`. Update entry point to `mosaicx = "mosaicx.cli:cli"`.

**Step 2: Install updated dependencies**

Run: `cd /Users/nutellabear/Documents/00-Code/MOSAICX && uv sync`
Expected: Successful install with new deps

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: update deps for v2 — DSPy replaces Instructor/Outlines/OpenAI"
```

---

### Task 2: Create MosaicxConfig with Pydantic Settings

**Files:**
- Create: `mosaicx/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for MosaicxConfig — Pydantic Settings single source of truth."""

import os
import pytest
from pathlib import Path


class TestMosaicxConfig:
    """Test MosaicxConfig defaults and overrides."""

    def test_default_values(self):
        """Config should have sensible defaults without any env vars."""
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig()
        assert cfg.lm == "ollama_chat/llama3.1:70b"
        assert cfg.lm_cheap == "ollama_chat/llama3.2:3b"
        assert cfg.completeness_threshold == 0.7
        assert cfg.batch_workers == 4
        assert cfg.checkpoint_every == 50
        assert cfg.default_template == "auto"
        assert cfg.deidentify_mode == "remove"
        assert cfg.default_export_formats == ["parquet", "jsonl"]
        assert cfg.force_ocr is False
        assert cfg.ocr_langs == ["en", "de"]

    def test_env_override(self, monkeypatch):
        """Environment variables with MOSAICX_ prefix override defaults."""
        from mosaicx.config import MosaicxConfig

        monkeypatch.setenv("MOSAICX_LM", "openai/gpt-4o")
        monkeypatch.setenv("MOSAICX_BATCH_WORKERS", "16")
        cfg = MosaicxConfig()
        assert cfg.lm == "openai/gpt-4o"
        assert cfg.batch_workers == 16

    def test_home_dir_default(self):
        """home_dir defaults to ~/.mosaicx."""
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig()
        assert cfg.home_dir == Path.home() / ".mosaicx"

    def test_derived_paths(self):
        """schema_dir, optimized_dir, etc. derive from home_dir."""
        from mosaicx.config import MosaicxConfig

        cfg = MosaicxConfig()
        assert cfg.schema_dir == cfg.home_dir / "schemas"
        assert cfg.optimized_dir == cfg.home_dir / "optimized"
        assert cfg.checkpoint_dir == cfg.home_dir / "checkpoints"
        assert cfg.log_dir == cfg.home_dir / "logs"

    def test_get_config_singleton(self):
        """get_config() returns the same instance."""
        from mosaicx.config import get_config

        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -x -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mosaicx.config'` (since the old config is at `mosaicx/utils/config.py` and has different contents)

**Step 3: Write implementation**

```python
# mosaicx/config.py
"""
MOSAICX Configuration — Single source of truth via Pydantic Settings.

Resolution order: CLI flags > env vars (MOSAICX_*) > config file > defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MosaicxConfig(BaseSettings):
    """Central configuration for MOSAICX v2."""

    model_config = SettingsConfigDict(
        env_prefix="MOSAICX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    lm: str = "ollama_chat/llama3.1:70b"
    lm_cheap: str = "ollama_chat/llama3.2:3b"
    api_key: str = "ollama"

    # --- Processing ---
    default_template: str = "auto"
    completeness_threshold: float = 0.7
    batch_workers: int = 4
    checkpoint_every: int = 50

    # --- Paths ---
    home_dir: Path = Field(default_factory=lambda: Path.home() / ".mosaicx")

    @property
    def schema_dir(self) -> Path:
        return self.home_dir / "schemas"

    @property
    def optimized_dir(self) -> Path:
        return self.home_dir / "optimized"

    @property
    def checkpoint_dir(self) -> Path:
        return self.home_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.home_dir / "logs"

    # --- De-identification ---
    deidentify_mode: Literal["remove", "pseudonymize", "dateshift"] = "remove"

    # --- Export ---
    default_export_formats: list[str] = Field(
        default_factory=lambda: ["parquet", "jsonl"]
    )

    # --- Document loading ---
    force_ocr: bool = False
    ocr_langs: list[str] = Field(default_factory=lambda: ["en", "de"])
    vlm_model: str = "gemma3:27b"


@lru_cache(maxsize=1)
def get_config() -> MosaicxConfig:
    """Return the global config singleton."""
    return MosaicxConfig()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -x -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add mosaicx/config.py tests/test_config.py
git commit -m "feat: add MosaicxConfig with Pydantic Settings"
```

---

### Task 3: Create document loader module

**Files:**
- Create: `mosaicx/documents/__init__.py`
- Create: `mosaicx/documents/loader.py`
- Create: `tests/test_documents.py`

**Step 1: Write the failing test**

```python
# tests/test_documents.py
"""Tests for the document loading module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestLoadedDocument:
    """Test the LoadedDocument dataclass."""

    def test_construction(self):
        from mosaicx.documents.loader import LoadedDocument

        doc = LoadedDocument(
            text="Hello world",
            source_path=Path("/tmp/test.pdf"),
            format="pdf",
            page_count=3,
        )
        assert doc.text == "Hello world"
        assert doc.format == "pdf"
        assert doc.page_count == 3

    def test_char_count(self):
        from mosaicx.documents.loader import LoadedDocument

        doc = LoadedDocument(text="abc", source_path=Path("/tmp/x.pdf"), format="pdf")
        assert doc.char_count == 3

    def test_is_empty(self):
        from mosaicx.documents.loader import LoadedDocument

        empty = LoadedDocument(text="", source_path=Path("/tmp/x.pdf"), format="pdf")
        nonempty = LoadedDocument(
            text="content", source_path=Path("/tmp/x.pdf"), format="pdf"
        )
        assert empty.is_empty is True
        assert nonempty.is_empty is False


class TestLoadDocument:
    """Test load_document function."""

    def test_unsupported_format_raises(self, tmp_path):
        from mosaicx.documents.loader import load_document

        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(bad_file)

    def test_missing_file_raises(self):
        from mosaicx.documents.loader import load_document

        with pytest.raises(FileNotFoundError):
            load_document(Path("/nonexistent/file.pdf"))

    def test_plain_text_loading(self, tmp_path):
        from mosaicx.documents.loader import load_document

        txt_file = tmp_path / "report.txt"
        txt_file.write_text("Patient presents with cough.")
        doc = load_document(txt_file)
        assert "Patient presents with cough" in doc.text
        assert doc.format == "txt"

    def test_markdown_loading(self, tmp_path):
        from mosaicx.documents.loader import load_document

        md_file = tmp_path / "report.md"
        md_file.write_text("# Findings\n\nNormal chest.")
        doc = load_document(md_file)
        assert "Normal chest" in doc.text
        assert doc.format == "md"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_documents.py -x -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# mosaicx/documents/__init__.py
"""Document loading — Docling wrapper for PDF/DOCX/PPTX + plain text."""

from .loader import LoadedDocument, load_document

__all__ = ["LoadedDocument", "load_document"]
```

```python
# mosaicx/documents/loader.py
"""
Unified document loading — converts PDF, DOCX, PPTX, Markdown, and plain text
into a LoadedDocument with extracted text.

Uses Docling for structured formats; falls back to plain read for .txt/.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None  # type: ignore[assignment,misc]

# Formats that Docling handles
_DOCLING_FORMATS = {".pdf", ".docx", ".pptx"}
# Formats we read directly
_TEXT_FORMATS = {".txt", ".md", ".markdown"}
_ALL_SUPPORTED = _DOCLING_FORMATS | _TEXT_FORMATS


@dataclass
class LoadedDocument:
    """A document converted to plain text."""

    text: str
    source_path: Path
    format: str
    page_count: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


def load_document(path: Path) -> LoadedDocument:
    """Load a document from disk and convert to text.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file format is unsupported.
        RuntimeError: If Docling is required but not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _ALL_SUPPORTED:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {sorted(_ALL_SUPPORTED)}"
        )

    if suffix in _TEXT_FORMATS:
        return _load_text(path, suffix.lstrip("."))

    return _load_with_docling(path, suffix.lstrip("."))


def _load_text(path: Path, fmt: str) -> LoadedDocument:
    """Load plain text / markdown files directly."""
    text = path.read_text(encoding="utf-8")
    return LoadedDocument(text=text, source_path=path, format=fmt)


def _load_with_docling(path: Path, fmt: str) -> LoadedDocument:
    """Load structured documents via Docling."""
    if DocumentConverter is None:
        raise RuntimeError(
            "Docling is required for PDF/DOCX/PPTX loading. "
            "Install with: pip install docling"
        )
    converter = DocumentConverter()
    result = converter.convert(str(path))
    text = result.document.export_to_markdown()
    page_count = getattr(result.document, "page_count", None)
    return LoadedDocument(
        text=text, source_path=path, format=fmt, page_count=page_count
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_documents.py -x -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add mosaicx/documents/ tests/test_documents.py
git commit -m "feat: add documents module with LoadedDocument and load_document"
```

---

### Task 4: Create CLI skeleton with Click groups

**Files:**
- Create: `mosaicx/cli.py` (new — replaces `mosaicx/cli/app.py` and `mosaicx/mosaicx.py`)
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
"""Tests for the CLI skeleton."""

import pytest
from click.testing import CliRunner


class TestCLISkeleton:
    """Test that CLI groups and subcommands are registered."""

    def test_cli_group_exists(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MOSAICX" in result.output or "mosaicx" in result.output.lower()

    def test_extract_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "document" in result.output.lower()

    def test_batch_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])
        assert result.exit_code == 0

    def test_template_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["template", "--help"])
        assert result.exit_code == 0

    def test_schema_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["schema", "--help"])
        assert result.exit_code == 0

    def test_summarize_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["summarize", "--help"])
        assert result.exit_code == 0

    def test_deidentify_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["deidentify", "--help"])
        assert result.exit_code == 0

    def test_optimize_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["optimize", "--help"])
        assert result.exit_code == 0

    def test_config_command_registered(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0

    def test_version_flag(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "2.0" in result.output or "mosaicx" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -x -v`
Expected: FAIL — the new `mosaicx.cli` module doesn't exist with the expected structure yet

**Step 3: Write implementation**

Create `mosaicx/cli.py` with Click group skeleton. Each subcommand is a stub that prints a "not yet implemented" message — actual logic will be wired in later phases. Use `rich.console.Console` for output.

```python
# mosaicx/cli.py
"""
MOSAICX CLI — thin Click wrapper over the Python API.

All commands are stubs during Phase 1; wired to real pipelines in later phases.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


def _version_callback(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    from mosaicx.config import get_config  # noqa: avoid circular
    click.echo("mosaicx 2.0.0a1")
    ctx.exit()


@click.group()
@click.option("--version", is_flag=True, callback=_version_callback,
              expose_value=False, is_eager=True, help="Show version.")
def cli() -> None:
    """MOSAICX — Medical Document Structuring Platform."""


# ── extract ──────────────────────────────────────────────────────────
@cli.command()
@click.option("--document", "-d", type=click.Path(exists=True, path_type=Path),
              required=True, help="Path to document (PDF/DOCX/TXT).")
@click.option("--template", "-t", default="auto", help="Template name or 'auto'.")
@click.option("--optimized", type=click.Path(exists=True, path_type=Path),
              default=None, help="Path to optimized pipeline JSON.")
def extract(document: Path, template: str, optimized: Path | None) -> None:
    """Extract structured data from a document."""
    console.print(f"[bold]extract[/bold] {document} template={template}")


# ── batch ────────────────────────────────────────────────────────────
@cli.command()
@click.option("--input-dir", "-i", type=click.Path(exists=True, path_type=Path),
              required=True, help="Directory of documents.")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path),
              required=True, help="Output directory.")
@click.option("--template", "-t", default="auto")
@click.option("--format", "formats", multiple=True, default=["parquet", "jsonl"])
@click.option("--workers", type=int, default=4)
@click.option("--completeness-threshold", type=float, default=0.7)
@click.option("--resume", type=str, default=None, help="Resume batch ID.")
def batch(input_dir: Path, output_dir: Path, template: str,
          formats: tuple[str, ...], workers: int,
          completeness_threshold: float, resume: str | None) -> None:
    """Batch-process a directory of documents."""
    console.print(f"[bold]batch[/bold] {input_dir} → {output_dir}")


# ── template ─────────────────────────────────────────────────────────
@cli.group()
def template() -> None:
    """Manage extraction templates."""


@template.command("create")
@click.argument("yaml_path", type=click.Path(exists=True, path_type=Path))
def template_create(yaml_path: Path) -> None:
    """Register a custom YAML template."""
    console.print(f"[bold]template create[/bold] {yaml_path}")


@template.command("list")
def template_list() -> None:
    """List available templates."""
    console.print("[bold]template list[/bold]")


@template.command("validate")
@click.argument("yaml_path", type=click.Path(exists=True, path_type=Path))
def template_validate(yaml_path: Path) -> None:
    """Validate a YAML template."""
    console.print(f"[bold]template validate[/bold] {yaml_path}")


# ── schema ───────────────────────────────────────────────────────────
@cli.group()
def schema() -> None:
    """Generate and manage extraction schemas."""


@schema.command("generate")
@click.option("--desc", "-d", required=True, help="Natural-language description.")
def schema_generate(desc: str) -> None:
    """Generate a schema from a description."""
    console.print(f"[bold]schema generate[/bold] '{desc}'")


@schema.command("list")
def schema_list() -> None:
    """List available schemas."""
    console.print("[bold]schema list[/bold]")


@schema.command("refine")
@click.argument("yaml_path", type=click.Path(exists=True, path_type=Path))
@click.option("--instruction", "-i", required=True)
def schema_refine(yaml_path: Path, instruction: str) -> None:
    """Refine an existing schema."""
    console.print(f"[bold]schema refine[/bold] {yaml_path}")


# ── summarize ────────────────────────────────────────────────────────
@cli.command()
@click.option("--dir", "report_dir", type=click.Path(exists=True, path_type=Path),
              required=True, help="Directory of patient reports.")
@click.option("--patient", required=True, help="Patient identifier.")
@click.option("--format", "formats", multiple=True, default=["json"])
def summarize(report_dir: Path, patient: str, formats: tuple[str, ...]) -> None:
    """Generate a longitudinal patient summary."""
    console.print(f"[bold]summarize[/bold] {report_dir} patient={patient}")


# ── deidentify ───────────────────────────────────────────────────────
@cli.command()
@click.option("--document", "-d", type=click.Path(exists=True, path_type=Path),
              default=None, help="Single document to de-identify.")
@click.option("--dir", "doc_dir", type=click.Path(exists=True, path_type=Path),
              default=None, help="Directory of documents.")
@click.option("--mode", type=click.Choice(["remove", "pseudonymize", "dateshift"]),
              default="remove")
@click.option("--workers", type=int, default=4)
def deidentify(document: Path | None, doc_dir: Path | None,
               mode: str, workers: int) -> None:
    """Remove or pseudonymize PHI from documents."""
    target = document or doc_dir
    console.print(f"[bold]deidentify[/bold] {target} mode={mode}")


# ── optimize ─────────────────────────────────────────────────────────
@cli.command()
@click.option("--pipeline", "-p", required=True,
              type=click.Choice(["radiology", "extraction", "deidentify", "summarize"]))
@click.option("--trainset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--valset", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--budget", type=click.Choice(["light", "medium", "heavy"]),
              default="medium")
@click.option("--save", type=click.Path(path_type=Path), required=True)
def optimize(pipeline: str, trainset: Path, valset: Path,
             budget: str, save: Path) -> None:
    """Optimize a pipeline with GEPA/MIPROv2."""
    console.print(f"[bold]optimize[/bold] {pipeline} budget={budget}")


# ── config ───────────────────────────────────────────────────────────
@cli.group()
def config() -> None:
    """View and update configuration."""


@config.command("show")
def config_show() -> None:
    """Show current configuration."""
    from mosaicx.config import get_config
    cfg = get_config()
    for k, v in cfg.model_dump().items():
        console.print(f"  {k}: {v}")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value."""
    console.print(f"[bold]config set[/bold] {key}={value}")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -x -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add mosaicx/cli.py tests/test_cli.py
git commit -m "feat: add CLI skeleton with all v2 command groups"
```

---

### Task 5: Update package __init__.py and display.py

**Files:**
- Modify: `mosaicx/__init__.py`
- Modify: `mosaicx/display.py` (simplify — remove cfonts dependency)

**Step 1: Write the failing test**

```python
# tests/test_package.py
"""Tests for top-level package API."""

import pytest


class TestPackageImports:
    """Verify the public API surface."""

    def test_version(self):
        import mosaicx
        assert hasattr(mosaicx, "__version__")
        assert "2.0" in mosaicx.__version__

    def test_config_importable(self):
        from mosaicx.config import MosaicxConfig, get_config
        assert MosaicxConfig is not None
        assert callable(get_config)

    def test_cli_importable(self):
        from mosaicx.cli import cli
        assert callable(cli)

    def test_documents_importable(self):
        from mosaicx.documents import LoadedDocument, load_document
        assert LoadedDocument is not None
        assert callable(load_document)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_package.py -x -v`
Expected: FAIL — version still shows 1.5.0, imports may break

**Step 3: Simplify display.py**

Remove the `cfonts` dependency. Keep `rich` console, banner via Rich panel, styled messages. Read the full current `display.py` first and simplify — keep `console`, `show_main_banner()`, and `styled_message()`. Remove everything else.

**Step 4: Update __init__.py**

```python
# mosaicx/__init__.py
"""
MOSAICX — Medical Document Structuring Platform.

Public API:
    - mosaicx.config.MosaicxConfig / get_config()
    - mosaicx.documents.load_document()
    - mosaicx.cli.cli (Click entry point)
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("mosaicx")
except PackageNotFoundError:
    __version__ = "2.0.0a1"

__all__ = ["__version__"]
```

Note: The old public API (`extract_pdf`, `generate_schema`, etc.) is removed. New API functions will be added as pipelines are implemented in later phases.

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_package.py -x -v`
Expected: All 4 tests PASS

**Step 6: Commit**

```bash
git add mosaicx/__init__.py mosaicx/display.py tests/test_package.py
git commit -m "feat: update package init and simplify display (drop cfonts)"
```

---

## Phase 2: DSPy Pipelines

### Task 6: Schema generator pipeline — safe SchemaSpec approach

**Files:**
- Create: `mosaicx/pipelines/__init__.py`
- Create: `mosaicx/pipelines/schema_gen.py`
- Create: `tests/test_schema_gen.py`

**Step 1: Write the failing test**

```python
# tests/test_schema_gen.py
"""Tests for the schema generator pipeline."""

import pytest
from pydantic import BaseModel


class TestSchemaSpec:
    """Test SchemaSpec model itself."""

    def test_schema_spec_construction(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec

        spec = SchemaSpec(
            class_name="PatientRecord",
            description="A patient record",
            fields=[
                FieldSpec(name="name", type="str", description="Patient name", required=True),
                FieldSpec(name="age", type="int", description="Age in years", required=True),
                FieldSpec(name="diagnosis", type="str", description="Diagnosis", required=False),
            ],
        )
        assert spec.class_name == "PatientRecord"
        assert len(spec.fields) == 3

    def test_compile_to_model(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, compile_schema

        spec = SchemaSpec(
            class_name="TestModel",
            description="Test",
            fields=[
                FieldSpec(name="name", type="str", description="Name", required=True),
                FieldSpec(name="score", type="float", description="Score", required=False),
            ],
        )
        Model = compile_schema(spec)
        assert Model.__name__ == "TestModel"

        # Validate it works as a Pydantic model
        instance = Model(name="Alice")
        assert instance.name == "Alice"
        assert instance.score is None

    def test_compile_with_list_field(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, compile_schema

        spec = SchemaSpec(
            class_name="ReportModel",
            description="Report",
            fields=[
                FieldSpec(name="findings", type="list[str]", description="List of findings", required=True),
            ],
        )
        Model = compile_schema(spec)
        instance = Model(findings=["nodule", "effusion"])
        assert len(instance.findings) == 2

    def test_compile_with_enum_field(self):
        from mosaicx.pipelines.schema_gen import SchemaSpec, FieldSpec, compile_schema

        spec = SchemaSpec(
            class_name="GenderModel",
            description="Gender",
            fields=[
                FieldSpec(
                    name="gender",
                    type="enum",
                    description="Gender",
                    required=True,
                    enum_values=["male", "female", "other"],
                ),
            ],
        )
        Model = compile_schema(spec)
        instance = Model(gender="male")
        assert instance.gender == "male"


class TestSchemaGeneratorSignature:
    """Test the DSPy signature exists and has correct fields."""

    def test_signature_has_fields(self):
        from mosaicx.pipelines.schema_gen import GenerateSchemaSpec

        sig = GenerateSchemaSpec
        # Input fields
        assert "description" in sig.input_fields
        # Output fields
        assert "schema_spec" in sig.output_fields
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema_gen.py -x -v`
Expected: FAIL — module does not exist

**Step 3: Write implementation**

```python
# mosaicx/pipelines/__init__.py
"""DSPy pipeline modules for MOSAICX."""
```

```python
# mosaicx/pipelines/schema_gen.py
"""
Schema Generator — safe structured approach replacing exec()-based code gen.

LLM outputs a SchemaSpec → compile_schema() builds a Pydantic model
programmatically via pydantic.create_model(). No code generation, no exec().
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import dspy
from pydantic import BaseModel, Field, create_model


# ── Data models ──────────────────────────────────────────────────────

class FieldSpec(BaseModel):
    """Specification for a single field in a schema."""

    name: str
    type: str  # "str", "int", "float", "bool", "list[str]", "enum", etc.
    description: str = ""
    required: bool = True
    enum_values: Optional[list[str]] = None


class SchemaSpec(BaseModel):
    """A complete schema specification — JSON-serializable, no code."""

    class_name: str
    description: str = ""
    fields: list[FieldSpec]


# ── Type mapping ─────────────────────────────────────────────────────

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
}


def _resolve_type(spec: FieldSpec) -> type:
    """Map FieldSpec.type to a Python type."""
    t = spec.type.lower().strip()

    if t == "enum" and spec.enum_values:
        return Enum(f"{spec.name}_enum", {v: v for v in spec.enum_values})  # type: ignore[return-value]

    if t.startswith("list["):
        inner = t[5:-1].strip()
        inner_type = _TYPE_MAP.get(inner, str)
        return list[inner_type]  # type: ignore[valid-type]

    return _TYPE_MAP.get(t, str)


# ── Compiler ─────────────────────────────────────────────────────────

def compile_schema(spec: SchemaSpec) -> type[BaseModel]:
    """Compile a SchemaSpec into a Pydantic BaseModel class.

    Uses pydantic.create_model() — no exec(), no code injection risk.
    """
    field_definitions: dict[str, Any] = {}
    for f in spec.fields:
        python_type = _resolve_type(f)
        if f.required:
            field_definitions[f.name] = (python_type, Field(description=f.description))
        else:
            field_definitions[f.name] = (
                Optional[python_type],
                Field(default=None, description=f.description),
            )

    return create_model(spec.class_name, **field_definitions)  # type: ignore[call-overload]


# ── DSPy Signature ───────────────────────────────────────────────────

class GenerateSchemaSpec(dspy.Signature):
    """Generate a structured schema specification from a natural-language description.

    Given a description of what data to extract, produce a SchemaSpec with
    field names, types, and descriptions. Do NOT generate code.
    """

    description: str = dspy.InputField(desc="Natural-language description of the schema")
    example_text: str = dspy.InputField(
        desc="Optional example document text for context", default=""
    )
    schema_spec: SchemaSpec = dspy.OutputField(desc="The generated schema specification")


# ── Pipeline Module ──────────────────────────────────────────────────

class SchemaGenerator(dspy.Module):
    """Generate a Pydantic model from a natural-language description."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateSchemaSpec)

    def forward(self, description: str, example_text: str = "") -> dspy.Prediction:
        result = self.generate(description=description, example_text=example_text)
        model_class = compile_schema(result.schema_spec)
        return dspy.Prediction(
            schema_spec=result.schema_spec,
            model_class=model_class,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schema_gen.py -x -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/ tests/test_schema_gen.py
git commit -m "feat: add schema generator pipeline with safe compile_schema()"
```

---

### Task 7: Custom YAML template compiler

**Files:**
- Create: `mosaicx/schemas/__init__.py`
- Create: `mosaicx/schemas/template_compiler.py`
- Create: `tests/test_template_compiler.py`

**Step 1: Write the failing test**

```python
# tests/test_template_compiler.py
"""Tests for the YAML template compiler."""

import pytest
import yaml
from pathlib import Path
from pydantic import BaseModel


SAMPLE_TEMPLATE_YAML = """\
name: ChestCTReport
description: "Structured chest CT report"

sections:
  - name: indication
    type: str
    required: true
  - name: technique
    type: str
    required: true
  - name: findings
    type: list
    item:
      type: object
      fields:
        - name: category
          type: enum
          values: ["nodule", "lymphadenopathy", "effusion", "other"]
        - name: description
          type: str
        - name: size_mm
          type: float
          required: false
  - name: impression
    type: str
    required: true
"""


class TestTemplateCompiler:
    """Test YAML template → Pydantic model compilation."""

    def test_compile_from_yaml_string(self):
        from mosaicx.schemas.template_compiler import compile_template

        Model = compile_template(SAMPLE_TEMPLATE_YAML)
        assert Model.__name__ == "ChestCTReport"

    def test_compiled_model_has_fields(self):
        from mosaicx.schemas.template_compiler import compile_template

        Model = compile_template(SAMPLE_TEMPLATE_YAML)
        fields = set(Model.model_fields.keys())
        assert "indication" in fields
        assert "technique" in fields
        assert "findings" in fields
        assert "impression" in fields

    def test_compiled_model_validates(self):
        from mosaicx.schemas.template_compiler import compile_template

        Model = compile_template(SAMPLE_TEMPLATE_YAML)
        instance = Model(
            indication="Cough",
            technique="CT chest with contrast",
            findings=[{"category": "nodule", "description": "5mm RUL nodule"}],
            impression="Pulmonary nodule, recommend follow-up.",
        )
        assert instance.indication == "Cough"
        assert len(instance.findings) == 1

    def test_compile_from_file(self, tmp_path):
        from mosaicx.schemas.template_compiler import compile_template_file

        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(SAMPLE_TEMPLATE_YAML)
        Model = compile_template_file(yaml_file)
        assert Model.__name__ == "ChestCTReport"

    def test_required_field_validation(self):
        from mosaicx.schemas.template_compiler import compile_template

        Model = compile_template(SAMPLE_TEMPLATE_YAML)
        with pytest.raises(Exception):  # Pydantic ValidationError
            Model(indication="Cough")  # missing technique and impression

    def test_parse_template_metadata(self):
        from mosaicx.schemas.template_compiler import parse_template

        meta = parse_template(SAMPLE_TEMPLATE_YAML)
        assert meta.name == "ChestCTReport"
        assert meta.description == "Structured chest CT report"
        assert len(meta.sections) == 4
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_template_compiler.py -x -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# mosaicx/schemas/__init__.py
"""Schema management — template compiler, ontology resolution, FHIR export."""
```

```python
# mosaicx/schemas/template_compiler.py
"""
Template Compiler — YAML template → Pydantic model → DSPy signatures.

Users define extraction templates in YAML; the compiler converts them to
Pydantic models via create_model(). No exec(), no code injection.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, create_model


# ── Template metadata ────────────────────────────────────────────────

class SectionSpec(BaseModel):
    """One section in a template YAML."""
    name: str
    type: str
    required: bool = True
    description: str = ""
    values: Optional[list[str]] = None  # for enum type
    item: Optional[dict] = None  # for list type with nested object
    fields: Optional[list[dict]] = None  # for object type


class TemplateMeta(BaseModel):
    """Parsed template metadata."""
    name: str
    description: str = ""
    radreport_id: Optional[str] = None
    sections: list[SectionSpec]


def parse_template(yaml_content: str) -> TemplateMeta:
    """Parse YAML content into TemplateMeta."""
    data = yaml.safe_load(yaml_content)
    sections = [SectionSpec(**s) for s in data.get("sections", [])]
    return TemplateMeta(
        name=data["name"],
        description=data.get("description", ""),
        radreport_id=data.get("radreport_id"),
        sections=sections,
    )


# ── Type resolution ──────────────────────────────────────────────────

_SIMPLE_TYPES: dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
}


def _build_enum(name: str, values: list[str]) -> type:
    """Build a string Enum from a list of values."""
    return Enum(name, {v: v for v in values})  # type: ignore[return-value]


def _build_nested_object(parent_name: str, fields: list[dict]) -> type[BaseModel]:
    """Build a nested Pydantic model from a list of field dicts."""
    field_defs: dict[str, Any] = {}
    for f in fields:
        fname = f["name"]
        ftype_str = f.get("type", "str")
        freq = f.get("required", True)
        fdesc = f.get("description", "")

        if ftype_str == "enum" and "values" in f:
            python_type = _build_enum(f"{parent_name}_{fname}", f["values"])
        else:
            python_type = _SIMPLE_TYPES.get(ftype_str, str)

        if freq:
            field_defs[fname] = (python_type, Field(description=fdesc))
        else:
            field_defs[fname] = (Optional[python_type], Field(default=None, description=fdesc))

    return create_model(f"{parent_name}Item", **field_defs)  # type: ignore[call-overload]


def _resolve_section_type(section: SectionSpec, model_name: str) -> tuple[type, bool]:
    """Resolve a section spec to (python_type, is_required)."""
    t = section.type.lower().strip()

    if t == "enum" and section.values:
        return _build_enum(f"{model_name}_{section.name}", section.values), section.required

    if t == "list":
        if section.item and section.item.get("type") == "object":
            nested = _build_nested_object(
                f"{model_name}_{section.name}",
                section.item.get("fields", []),
            )
            return list[nested], section.required  # type: ignore[valid-type]
        return list[str], section.required  # type: ignore[valid-type]

    if t == "object" and section.fields:
        nested = _build_nested_object(f"{model_name}_{section.name}", [f for f in section.fields if isinstance(f, dict)])
        return nested, section.required

    return _SIMPLE_TYPES.get(t, str), section.required


# ── Public API ───────────────────────────────────────────────────────

def compile_template(yaml_content: str) -> type[BaseModel]:
    """Compile a YAML template string into a Pydantic BaseModel class."""
    meta = parse_template(yaml_content)
    field_defs: dict[str, Any] = {}

    for section in meta.sections:
        python_type, required = _resolve_section_type(section, meta.name)
        desc = section.description or section.name
        if required:
            field_defs[section.name] = (python_type, Field(description=desc))
        else:
            field_defs[section.name] = (
                Optional[python_type],
                Field(default=None, description=desc),
            )

    return create_model(meta.name, **field_defs)  # type: ignore[call-overload]


def compile_template_file(path: Path) -> type[BaseModel]:
    """Compile a YAML template file into a Pydantic BaseModel class."""
    content = Path(path).read_text(encoding="utf-8")
    return compile_template(content)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_template_compiler.py -x -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add mosaicx/schemas/ tests/test_template_compiler.py
git commit -m "feat: add YAML template compiler with safe create_model()"
```

---

### Task 8: Generic document extractor pipeline

**Files:**
- Create: `mosaicx/pipelines/extraction.py`
- Create: `tests/test_extraction_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_extraction_pipeline.py
"""Tests for the generic document extraction pipeline."""

import pytest
import dspy


class TestExtractionSignatures:
    """Test that DSPy signatures have correct input/output fields."""

    def test_extract_demographics_signature(self):
        from mosaicx.pipelines.extraction import ExtractDemographics

        assert "document_text" in ExtractDemographics.input_fields
        assert "demographics" in ExtractDemographics.output_fields

    def test_extract_findings_signature(self):
        from mosaicx.pipelines.extraction import ExtractFindings

        assert "document_text" in ExtractFindings.input_fields
        assert "findings" in ExtractFindings.output_fields

    def test_extract_diagnoses_signature(self):
        from mosaicx.pipelines.extraction import ExtractDiagnoses

        assert "document_text" in ExtractDiagnoses.input_fields
        assert "diagnoses" in ExtractDiagnoses.output_fields


class TestDocumentExtractorModule:
    """Test the DocumentExtractor DSPy module."""

    def test_module_has_submodules(self):
        from mosaicx.pipelines.extraction import DocumentExtractor

        extractor = DocumentExtractor()
        assert hasattr(extractor, "extract_demographics")
        assert hasattr(extractor, "extract_findings")
        assert hasattr(extractor, "extract_diagnoses")

    def test_module_accepts_custom_schema(self):
        from mosaicx.pipelines.extraction import DocumentExtractor
        from pydantic import BaseModel

        class CustomReport(BaseModel):
            summary: str
            category: str

        extractor = DocumentExtractor(output_schema=CustomReport)
        assert hasattr(extractor, "extract_custom")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_extraction_pipeline.py -x -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# mosaicx/pipelines/extraction.py
"""
Generic Document Extractor — 3-module DSPy chain.

1. ExtractDemographics: patient demographics
2. ExtractFindings: clinical findings (uses demographics as context)
3. ExtractDiagnoses: diagnoses (uses findings as context)

Also supports custom schemas via a dynamic signature.
"""

from __future__ import annotations

from typing import Optional

import dspy
from pydantic import BaseModel


# ── Output models ────────────────────────────────────────────────────

class Demographics(BaseModel):
    """Extracted patient demographics."""
    patient_id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    date_of_birth: Optional[str] = None


class Finding(BaseModel):
    """A single clinical finding."""
    description: str
    location: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None


class Diagnosis(BaseModel):
    """A single diagnosis."""
    name: str
    icd10_code: Optional[str] = None
    confidence: Optional[str] = None


# ── DSPy Signatures ──────────────────────────────────────────────────

class ExtractDemographics(dspy.Signature):
    """Extract patient demographics from a clinical document."""

    document_text: str = dspy.InputField(desc="Full text of the clinical document")
    demographics: Demographics = dspy.OutputField(desc="Extracted patient demographics")


class ExtractFindings(dspy.Signature):
    """Extract clinical findings from a document, given patient demographics context."""

    document_text: str = dspy.InputField(desc="Full text of the clinical document")
    demographics_context: str = dspy.InputField(
        desc="Previously extracted demographics as context", default=""
    )
    findings: list[Finding] = dspy.OutputField(desc="List of clinical findings")


class ExtractDiagnoses(dspy.Signature):
    """Extract diagnoses from a document, given findings context."""

    document_text: str = dspy.InputField(desc="Full text of the clinical document")
    findings_context: str = dspy.InputField(
        desc="Previously extracted findings as context", default=""
    )
    diagnoses: list[Diagnosis] = dspy.OutputField(desc="List of diagnoses")


# ── Pipeline Module ──────────────────────────────────────────────────

class DocumentExtractor(dspy.Module):
    """Generic 3-step document extraction pipeline.

    If output_schema is provided, uses a single dynamic signature
    instead of the 3-step chain.
    """

    def __init__(self, output_schema: Optional[type[BaseModel]] = None) -> None:
        super().__init__()

        if output_schema is not None:
            # Dynamic signature for custom schema
            self.extract_custom = dspy.ChainOfThought(
                dspy.Signature(
                    {
                        "document_text": dspy.InputField(desc="Full text of the document"),
                        "result": dspy.OutputField(
                            desc=f"Extracted data as {output_schema.__name__}",
                            type=output_schema,
                        ),
                    },
                    instructions=f"Extract structured data matching the {output_schema.__name__} schema.",
                )
            )
        else:
            self.extract_demographics = dspy.ChainOfThought(ExtractDemographics)
            self.extract_findings = dspy.ChainOfThought(ExtractFindings)
            self.extract_diagnoses = dspy.ChainOfThought(ExtractDiagnoses)

    def forward(self, document_text: str) -> dspy.Prediction:
        if hasattr(self, "extract_custom"):
            return self.extract_custom(document_text=document_text)

        demo_result = self.extract_demographics(document_text=document_text)
        demographics = demo_result.demographics

        findings_result = self.extract_findings(
            document_text=document_text,
            demographics_context=str(demographics),
        )
        findings = findings_result.findings

        diagnoses_result = self.extract_diagnoses(
            document_text=document_text,
            findings_context=str(findings),
        )

        return dspy.Prediction(
            demographics=demographics,
            findings=findings,
            diagnoses=diagnoses_result.diagnoses,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_extraction_pipeline.py -x -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add mosaicx/pipelines/extraction.py tests/test_extraction_pipeline.py
git commit -m "feat: add generic document extractor pipeline (3-step DSPy chain)"
```

---

### Task 9: Radiology report structurer pipeline (base models + signatures)

**Files:**
- Create: `mosaicx/schemas/radreport/__init__.py`
- Create: `mosaicx/schemas/radreport/base.py`
- Create: `mosaicx/pipelines/radiology.py`
- Create: `tests/test_radiology_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_radiology_pipeline.py
"""Tests for the radiology report structurer pipeline."""

import pytest


class TestRadReportBaseModels:
    """Test base data models for radiology reports."""

    def test_measurement_construction(self):
        from mosaicx.schemas.radreport.base import Measurement

        m = Measurement(value=5.2, unit="mm", dimension="diameter")
        assert m.value == 5.2
        assert m.unit == "mm"

    def test_finding_construction(self):
        from mosaicx.schemas.radreport.base import RadReportFinding

        f = RadReportFinding(
            anatomy="right upper lobe",
            observation="nodule",
            description="5mm ground glass nodule",
        )
        assert f.anatomy == "right upper lobe"
        assert f.radlex_id is None  # optional

    def test_impression_item(self):
        from mosaicx.schemas.radreport.base import ImpressionItem

        imp = ImpressionItem(
            statement="Pulmonary nodule, recommend CT follow-up in 12 months.",
            category="Lung-RADS 3",
            actionable=True,
        )
        assert imp.actionable is True

    def test_change_type(self):
        from mosaicx.schemas.radreport.base import ChangeType

        c = ChangeType(status="increased", prior_date="2025-06-15")
        assert c.status == "increased"


class TestRadiologyPipelineSignatures:
    """Test DSPy signatures for the radiology pipeline."""

    def test_classify_exam_type_signature(self):
        from mosaicx.pipelines.radiology import ClassifyExamType

        assert "report_header" in ClassifyExamType.input_fields
        assert "exam_type" in ClassifyExamType.output_fields

    def test_parse_sections_signature(self):
        from mosaicx.pipelines.radiology import ParseReportSections

        assert "report_text" in ParseReportSections.input_fields
        assert "sections" in ParseReportSections.output_fields

    def test_extract_findings_signature(self):
        from mosaicx.pipelines.radiology import ExtractRadFindings

        assert "findings_text" in ExtractRadFindings.input_fields
        assert "findings" in ExtractRadFindings.output_fields

    def test_extract_impression_signature(self):
        from mosaicx.pipelines.radiology import ExtractImpression

        assert "impression_text" in ExtractImpression.input_fields
        assert "impressions" in ExtractImpression.output_fields


class TestRadiologyStructurerModule:
    """Test the RadiologyReportStructurer DSPy module."""

    def test_module_has_submodules(self):
        from mosaicx.pipelines.radiology import RadiologyReportStructurer

        pipeline = RadiologyReportStructurer()
        assert hasattr(pipeline, "classify_exam")
        assert hasattr(pipeline, "parse_sections")
        assert hasattr(pipeline, "extract_findings")
        assert hasattr(pipeline, "extract_impression")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_radiology_pipeline.py -x -v`
Expected: FAIL

**Step 3: Write implementation**

Create `mosaicx/schemas/radreport/__init__.py`, `mosaicx/schemas/radreport/base.py` with the base data models (RadReportFinding, Measurement, ChangeType, ImpressionItem, ReportSections), and `mosaicx/pipelines/radiology.py` with the 5-step DSPy pipeline.

`mosaicx/schemas/radreport/base.py`:

```python
"""Base data models shared by all RadReport templates."""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Measurement(BaseModel):
    """A radiological measurement."""
    value: float
    unit: str = "mm"
    dimension: str = ""
    prior_value: Optional[float] = None


class ChangeType(BaseModel):
    """Change from prior exam."""
    status: Literal["new", "stable", "increased", "decreased", "resolved"]
    prior_date: Optional[str] = None
    prior_measurement: Optional[Measurement] = None


class RadReportFinding(BaseModel):
    """A structured radiology finding."""
    anatomy: str
    radlex_id: Optional[str] = None
    observation: str = ""
    description: str = ""
    measurement: Optional[Measurement] = None
    change_from_prior: Optional[ChangeType] = None
    severity: Optional[str] = None
    template_field_id: Optional[str] = None


class ImpressionItem(BaseModel):
    """A structured impression entry."""
    statement: str
    category: Optional[str] = None  # BI-RADS, Lung-RADS, etc.
    icd10_code: Optional[str] = None
    actionable: bool = False
    finding_refs: list[int] = Field(default_factory=list)


class ReportSections(BaseModel):
    """Standard radiology report sections."""
    indication: str = ""
    comparison: str = ""
    technique: str = ""
    findings: str = ""
    impression: str = ""
```

`mosaicx/pipelines/radiology.py`:

```python
"""
Radiology Report Structurer — 5-step DSPy pipeline with RadReport template alignment.

Steps:
1. ClassifyExamType — route to correct template
2. ParseReportSections — segment into standard sections
3. ExtractTechnique — modality, body region, contrast, protocol
4. ExtractRadFindings — structured findings with anatomy, measurements
5. ExtractImpression — impression items with scoring systems
"""

from __future__ import annotations
from typing import Literal, Optional
import dspy
from pydantic import BaseModel
from mosaicx.schemas.radreport.base import (
    RadReportFinding, ImpressionItem, ReportSections,
)


# ── Signatures ───────────────────────────────────────────────────────

class ClassifyExamType(dspy.Signature):
    """Classify a radiology report into an exam type from the header/first 500 chars."""
    report_header: str = dspy.InputField(desc="First 500 characters of the report")
    exam_type: str = dspy.OutputField(
        desc="Exam type: chest_ct, chest_xr, brain_mri, abdomen_ct, mammography, "
             "thyroid_us, lung_ct, msk_mri, cardiac_mri, pet_ct, or generic"
    )


class ParseReportSections(dspy.Signature):
    """Parse a radiology report into standard sections."""
    report_text: str = dspy.InputField(desc="Full radiology report text")
    sections: ReportSections = dspy.OutputField(desc="Parsed report sections")


class ExtractTechnique(dspy.Signature):
    """Extract imaging technique details from the technique section."""
    technique_text: str = dspy.InputField(desc="Technique section text")
    modality: str = dspy.OutputField(desc="Imaging modality (CT, MRI, US, XR, PET)")
    body_region: str = dspy.OutputField(desc="Body region imaged")
    contrast: str = dspy.OutputField(desc="Contrast agent used, or 'none'")
    protocol: str = dspy.OutputField(desc="Protocol details, or 'standard'")


class ExtractRadFindings(dspy.Signature):
    """Extract structured findings from the findings section of a radiology report.

    Each finding should include anatomy, observation, measurement if present,
    and change from prior if applicable.
    """
    findings_text: str = dspy.InputField(desc="Findings section text")
    exam_type: str = dspy.InputField(desc="Exam type for template context")
    findings: list[RadReportFinding] = dspy.OutputField(
        desc="List of structured findings"
    )


class ExtractImpression(dspy.Signature):
    """Extract structured impression items with scoring system categories."""
    impression_text: str = dspy.InputField(desc="Impression section text")
    exam_type: str = dspy.InputField(desc="Exam type for scoring context")
    findings_context: str = dspy.InputField(desc="Previously extracted findings for cross-reference")
    impressions: list[ImpressionItem] = dspy.OutputField(
        desc="List of structured impression items"
    )


# ── Pipeline Module ──────────────────────────────────────────────────

class RadiologyReportStructurer(dspy.Module):
    """5-step radiology report structuring pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.classify_exam = dspy.Predict(ClassifyExamType)
        self.parse_sections = dspy.Predict(ParseReportSections)
        self.extract_technique = dspy.Predict(ExtractTechnique)
        self.extract_findings = dspy.ChainOfThought(ExtractRadFindings)
        self.extract_impression = dspy.ChainOfThought(ExtractImpression)

    def forward(self, report_text: str) -> dspy.Prediction:
        # Step 1: Classify exam type
        header = report_text[:500]
        classification = self.classify_exam(report_header=header)
        exam_type = classification.exam_type

        # Step 2: Parse sections
        parsed = self.parse_sections(report_text=report_text)
        sections = parsed.sections

        # Step 3: Extract technique
        technique = self.extract_technique(technique_text=sections.technique)

        # Step 4: Extract findings
        findings_result = self.extract_findings(
            findings_text=sections.findings,
            exam_type=exam_type,
        )

        # Step 5: Extract impressions
        impression_result = self.extract_impression(
            impression_text=sections.impression,
            exam_type=exam_type,
            findings_context=str(findings_result.findings),
        )

        return dspy.Prediction(
            exam_type=exam_type,
            sections=sections,
            modality=technique.modality,
            body_region=technique.body_region,
            contrast=technique.contrast,
            protocol=technique.protocol,
            findings=findings_result.findings,
            impressions=impression_result.impressions,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_radiology_pipeline.py -x -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add mosaicx/schemas/radreport/ mosaicx/pipelines/radiology.py tests/test_radiology_pipeline.py
git commit -m "feat: add radiology report structurer with 5-step DSPy pipeline"
```

---

### Task 10: Summarizer pipeline

**Files:**
- Create: `mosaicx/pipelines/summarizer.py`
- Create: `tests/test_summarizer_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_summarizer_pipeline.py
"""Tests for the report summarizer pipeline."""

import pytest


class TestSummarizerSignatures:
    def test_extract_timeline_event_signature(self):
        from mosaicx.pipelines.summarizer import ExtractTimelineEvent
        assert "report_text" in ExtractTimelineEvent.input_fields
        assert "event" in ExtractTimelineEvent.output_fields

    def test_synthesize_timeline_signature(self):
        from mosaicx.pipelines.summarizer import SynthesizeTimeline
        assert "events_json" in SynthesizeTimeline.input_fields
        assert "narrative" in SynthesizeTimeline.output_fields


class TestSummarizerModule:
    def test_module_has_submodules(self):
        from mosaicx.pipelines.summarizer import ReportSummarizer
        s = ReportSummarizer()
        assert hasattr(s, "extract_event")
        assert hasattr(s, "synthesize")

    def test_timeline_event_model(self):
        from mosaicx.pipelines.summarizer import TimelineEvent
        ev = TimelineEvent(
            date="2025-06-15",
            exam_type="CT chest",
            key_finding="5mm RUL nodule, new",
            clinical_context="Cough for 3 weeks",
        )
        assert ev.exam_type == "CT chest"
```

**Step 2: Run to verify failure, then implement**

`mosaicx/pipelines/summarizer.py` — two-step pipeline: ExtractTimelineEvent (per-report, parallelizable) → SynthesizeTimeline (weave into narrative).

**Step 3: Commit**

```bash
git add mosaicx/pipelines/summarizer.py tests/test_summarizer_pipeline.py
git commit -m "feat: add report summarizer pipeline with timeline synthesis"
```

---

### Task 11: Deidentifier pipeline

**Files:**
- Create: `mosaicx/pipelines/deidentifier.py`
- Create: `tests/test_deidentifier_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_deidentifier_pipeline.py
"""Tests for the de-identification pipeline."""

import pytest
import re


class TestPHIRegex:
    """Test the deterministic regex guard patterns."""

    def test_ssn_detection(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "SSN: 123-45-6789"
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert any(m for m in matches if m)

    def test_phone_detection(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "Phone: (555) 123-4567"
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert any(m for m in matches if m)

    def test_mrn_detection(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "MRN: 12345678"
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert any(m for m in matches if m)

    def test_clean_text_no_match(self):
        from mosaicx.pipelines.deidentifier import PHI_PATTERNS
        text = "The lungs are clear. No pleural effusion."
        matches = [p.search(text) for p in PHI_PATTERNS]
        assert not any(m for m in matches if m)


class TestDeidentifierSignature:
    def test_redact_phi_signature(self):
        from mosaicx.pipelines.deidentifier import RedactPHI
        assert "document_text" in RedactPHI.input_fields
        assert "redacted_text" in RedactPHI.output_fields


class TestDeidentifierModule:
    def test_module_has_submodules(self):
        from mosaicx.pipelines.deidentifier import Deidentifier
        d = Deidentifier()
        assert hasattr(d, "redact")

    def test_regex_scrub(self):
        from mosaicx.pipelines.deidentifier import regex_scrub_phi
        text = "Patient John Smith, SSN 123-45-6789, phone (555) 123-4567."
        scrubbed = regex_scrub_phi(text)
        assert "123-45-6789" not in scrubbed
        assert "(555) 123-4567" not in scrubbed
```

**Step 2: Run to verify failure, implement, verify passes**

Create `mosaicx/pipelines/deidentifier.py` with:
- `PHI_PATTERNS`: compiled regex list for SSN, phone, MRN, dates, email
- `regex_scrub_phi(text)`: deterministic final guard
- `RedactPHI` DSPy signature
- `Deidentifier` module with LLM redaction + regex guard

**Step 3: Commit**

```bash
git add mosaicx/pipelines/deidentifier.py tests/test_deidentifier_pipeline.py
git commit -m "feat: add deidentifier pipeline with LLM + regex belt-and-suspenders"
```

---

## Phase 3: RadReport Templates & Auto-Selection

### Task 12: RadReport template registry and auto-selection

**Files:**
- Create: `mosaicx/schemas/radreport/registry.py`
- Create: `tests/test_radreport_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_radreport_registry.py
"""Tests for RadReport template registry and auto-selection."""

import pytest


class TestTemplateRegistry:
    def test_list_templates(self):
        from mosaicx.schemas.radreport.registry import list_templates
        templates = list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_get_template_by_name(self):
        from mosaicx.schemas.radreport.registry import get_template
        t = get_template("generic")
        assert t is not None
        assert t.name == "generic"

    def test_get_template_unknown_returns_none(self):
        from mosaicx.schemas.radreport.registry import get_template
        t = get_template("nonexistent_modality_xyz")
        assert t is None

    def test_template_has_exam_type(self):
        from mosaicx.schemas.radreport.registry import list_templates
        for t in list_templates():
            assert hasattr(t, "exam_type")
            assert isinstance(t.exam_type, str)
```

**Step 2–5: Implement, test, commit**

Create `mosaicx/schemas/radreport/registry.py` with a `TemplateInfo` dataclass (name, exam_type, radreport_id, description, output_model) and `list_templates()` / `get_template(name)` functions. Start with a `generic` template and `chest_ct` as the first real template.

```bash
git commit -m "feat: add RadReport template registry with auto-selection"
```

---

### Task 13: Scoring systems (BI-RADS, Lung-RADS, TI-RADS)

**Files:**
- Create: `mosaicx/schemas/radreport/scoring.py`
- Create: `tests/test_scoring.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring.py
"""Tests for radiology scoring systems."""

import pytest


class TestScoringModels:
    def test_lung_rads_values(self):
        from mosaicx.schemas.radreport.scoring import LungRADS
        assert LungRADS.CATEGORY_1.value == "1"
        assert LungRADS.CATEGORY_4B.value == "4B"

    def test_birads_values(self):
        from mosaicx.schemas.radreport.scoring import BIRADS
        assert BIRADS.CATEGORY_0.value == "0"
        assert BIRADS.CATEGORY_6.value == "6"

    def test_tirads_values(self):
        from mosaicx.schemas.radreport.scoring import TIRADS
        assert TIRADS.TR1.value == "TR1"

    def test_deauville_values(self):
        from mosaicx.schemas.radreport.scoring import Deauville
        assert Deauville.SCORE_1.value == "1"
        assert Deauville.SCORE_5.value == "5"

    def test_scoring_descriptions(self):
        from mosaicx.schemas.radreport.scoring import get_scoring_description
        desc = get_scoring_description("Lung-RADS", "4B")
        assert isinstance(desc, str)
        assert len(desc) > 0
```

**Step 2–5: Implement scoring enums, test, commit**

```bash
git commit -m "feat: add scoring system enums (BI-RADS, Lung-RADS, TI-RADS, Deauville)"
```

---

## Phase 4: Quality & Ontology

### Task 14: Completeness evaluator — field coverage (Layer 1)

**Files:**
- Create: `mosaicx/evaluation/__init__.py`
- Create: `mosaicx/evaluation/completeness.py`
- Create: `tests/test_completeness.py`

**Step 1: Write the failing test**

```python
# tests/test_completeness.py
"""Tests for the completeness evaluator."""

import pytest
from pydantic import BaseModel, Field
from typing import Optional


class SampleReport(BaseModel):
    indication: str
    findings: list[str]
    impression: str
    technique: Optional[str] = None


class TestFieldCoverage:
    """Test Layer 1 — deterministic field coverage."""

    def test_all_fields_populated(self):
        from mosaicx.evaluation.completeness import field_coverage

        report = SampleReport(
            indication="Cough",
            findings=["nodule"],
            impression="Follow-up recommended",
            technique="CT with contrast",
        )
        score = field_coverage(report)
        assert score == 1.0

    def test_optional_field_missing(self):
        from mosaicx.evaluation.completeness import field_coverage

        report = SampleReport(
            indication="Cough",
            findings=["nodule"],
            impression="Follow-up",
        )
        # technique is optional and missing — still counts against coverage
        score = field_coverage(report)
        assert 0.5 < score < 1.0

    def test_empty_list_penalized(self):
        from mosaicx.evaluation.completeness import field_coverage

        report = SampleReport(
            indication="Cough",
            findings=[],  # empty list
            impression="Normal",
        )
        score = field_coverage(report)
        assert score < 1.0

    def test_empty_string_penalized(self):
        from mosaicx.evaluation.completeness import field_coverage

        report = SampleReport(
            indication="",
            findings=["nodule"],
            impression="Normal",
        )
        score = field_coverage(report)
        assert score < 1.0


class TestInformationDensity:
    """Test Layer 3 — information density."""

    def test_density_ratio(self):
        from mosaicx.evaluation.completeness import information_density

        source = "This is a very long report with many findings and detailed descriptions of all anatomical structures."
        structured = SampleReport(
            indication="Report",
            findings=["finding1"],
            impression="Normal",
        )
        density = information_density(source, structured)
        assert 0.0 <= density <= 1.0

    def test_density_empty_source(self):
        from mosaicx.evaluation.completeness import information_density

        structured = SampleReport(
            indication="x",
            findings=["x"],
            impression="x",
        )
        density = information_density("", structured)
        assert density == 0.0


class TestCompletenessScore:
    """Test the overall completeness score aggregation."""

    def test_overall_score_structure(self):
        from mosaicx.evaluation.completeness import compute_completeness

        report = SampleReport(
            indication="Cough",
            findings=["5mm nodule"],
            impression="Follow-up",
            technique="CT",
        )
        result = compute_completeness(report, source_text="Patient presents with cough. CT chest with contrast. 5mm nodule in RUL. Follow-up recommended.")
        assert "overall" in result
        assert "field_coverage" in result
        assert "information_density" in result
        assert 0.0 <= result["overall"] <= 1.0
```

**Step 2: Run to verify failure, implement, verify passes**

Create `mosaicx/evaluation/completeness.py` with:
- `field_coverage(model: BaseModel) -> float`
- `information_density(source: str, structured: BaseModel) -> float`
- `compute_completeness(structured, source_text) -> dict`

Layer 2 (semantic completeness via LLM) will be added as an optional parameter — skip for now.

**Step 3: Commit**

```bash
git add mosaicx/evaluation/ tests/test_completeness.py
git commit -m "feat: add completeness evaluator (field coverage + info density)"
```

---

### Task 15: Metrics and reward functions

**Files:**
- Create: `mosaicx/evaluation/metrics.py`
- Create: `mosaicx/evaluation/rewards.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_metrics.py
"""Tests for evaluation metrics and reward functions."""

import pytest


class TestExtractionReward:
    def test_penalizes_empty_findings(self):
        from mosaicx.evaluation.rewards import extraction_reward
        score = extraction_reward(findings=[], impression="Normal")
        assert score < 0.5

    def test_rewards_complete_extraction(self):
        from mosaicx.evaluation.rewards import extraction_reward
        score = extraction_reward(
            findings=[{"anatomy": "RUL", "observation": "nodule", "description": "5mm"}],
            impression="Pulmonary nodule, follow-up.",
        )
        assert score > 0.5

    def test_penalizes_findings_without_anatomy(self):
        from mosaicx.evaluation.rewards import extraction_reward
        score = extraction_reward(
            findings=[{"anatomy": "", "observation": "nodule", "description": "something"}],
            impression="Normal",
        )
        low = score
        score2 = extraction_reward(
            findings=[{"anatomy": "RUL", "observation": "nodule", "description": "something"}],
            impression="Normal",
        )
        assert score2 > low


class TestPHILeakReward:
    def test_clean_text_scores_high(self):
        from mosaicx.evaluation.rewards import phi_leak_reward
        score = phi_leak_reward("The lungs are clear bilaterally.")
        assert score == 1.0

    def test_text_with_ssn_scores_zero(self):
        from mosaicx.evaluation.rewards import phi_leak_reward
        score = phi_leak_reward("Patient SSN 123-45-6789.")
        assert score == 0.0

    def test_text_with_phone_scores_zero(self):
        from mosaicx.evaluation.rewards import phi_leak_reward
        score = phi_leak_reward("Call (555) 123-4567.")
        assert score == 0.0
```

**Step 2–5: Implement, test, commit**

```bash
git commit -m "feat: add extraction reward and PHI leak reward functions"
```

---

### Task 16: Ontology resolver (local lookup)

**Files:**
- Create: `mosaicx/schemas/ontology.py`
- Create: `tests/test_ontology.py`

**Step 1: Write the failing test**

```python
# tests/test_ontology.py
"""Tests for the ontology resolver."""

import pytest


class TestOntologyResolver:
    def test_exact_match_radlex(self):
        from mosaicx.schemas.ontology import OntologyResolver
        resolver = OntologyResolver()
        result = resolver.resolve("right upper lobe", vocabulary="radlex")
        assert result is not None
        assert result.term is not None

    def test_unknown_term_returns_none(self):
        from mosaicx.schemas.ontology import OntologyResolver
        resolver = OntologyResolver()
        result = resolver.resolve("xyzzy_nonexistent_term", vocabulary="radlex")
        assert result is None or result.confidence < 0.5

    def test_resolve_returns_code(self):
        from mosaicx.schemas.ontology import OntologyResult
        r = OntologyResult(term="right upper lobe", code="RID1303", vocabulary="radlex", confidence=0.95)
        assert r.code == "RID1303"

    def test_supported_vocabularies(self):
        from mosaicx.schemas.ontology import OntologyResolver
        resolver = OntologyResolver()
        vocabs = resolver.supported_vocabularies
        assert "radlex" in vocabs
```

**Step 2–5: Implement, test, commit**

Create `mosaicx/schemas/ontology.py` with `OntologyResult` model and `OntologyResolver` class. Initially uses a small built-in lookup dict for the most common RadLex terms (50-100 entries). The full terminology tables (compressed JSON) will be added as a data-loading enhancement later — for now the resolver works with a minimal embedded dict and gracefully returns None for unknown terms.

```bash
git commit -m "feat: add ontology resolver with local RadLex lookup"
```

---

### Task 17: FHIR R4 export

**Files:**
- Create: `mosaicx/schemas/fhir.py`
- Create: `tests/test_fhir.py`

**Step 1: Write the failing test**

```python
# tests/test_fhir.py
"""Tests for FHIR R4 DiagnosticReport bundle construction."""

import pytest
import json


class TestFHIRBundle:
    def test_build_diagnostic_report(self):
        from mosaicx.schemas.fhir import build_diagnostic_report
        bundle = build_diagnostic_report(
            patient_id="P001",
            findings=[{"anatomy": "right upper lobe", "observation": "nodule"}],
            impression="Pulmonary nodule.",
            procedure_code="24627-2",  # LOINC for CT Chest
        )
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        entries = bundle["entry"]
        resource_types = [e["resource"]["resourceType"] for e in entries]
        assert "DiagnosticReport" in resource_types
        assert "Observation" in resource_types

    def test_bundle_is_valid_json(self):
        from mosaicx.schemas.fhir import build_diagnostic_report
        bundle = build_diagnostic_report(
            patient_id="P001",
            findings=[],
            impression="Normal.",
        )
        # Should be serializable
        json_str = json.dumps(bundle)
        assert len(json_str) > 0

    def test_observation_per_finding(self):
        from mosaicx.schemas.fhir import build_diagnostic_report
        bundle = build_diagnostic_report(
            patient_id="P001",
            findings=[
                {"anatomy": "RUL", "observation": "nodule"},
                {"anatomy": "RLL", "observation": "atelectasis"},
            ],
            impression="Two findings.",
        )
        observations = [
            e for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Observation"
        ]
        assert len(observations) == 2
```

**Step 2–5: Implement, test, commit**

```bash
git commit -m "feat: add FHIR R4 DiagnosticReport bundle builder"
```

---

## Phase 5: Optimization & Batch

### Task 18: Optimization workflow module

**Files:**
- Create: `mosaicx/evaluation/optimize.py`
- Create: `tests/test_optimize.py`

**Step 1: Write the failing test**

```python
# tests/test_optimize.py
"""Tests for the optimization workflow."""

import pytest


class TestOptimizationConfig:
    def test_budget_presets(self):
        from mosaicx.evaluation.optimize import get_optimizer_config
        light = get_optimizer_config("light")
        medium = get_optimizer_config("medium")
        heavy = get_optimizer_config("heavy")
        assert light["max_iterations"] < medium["max_iterations"] < heavy["max_iterations"]

    def test_progressive_strategy(self):
        from mosaicx.evaluation.optimize import OPTIMIZATION_STRATEGY
        assert len(OPTIMIZATION_STRATEGY) == 3
        names = [s["name"] for s in OPTIMIZATION_STRATEGY]
        assert "BootstrapFewShot" in names
        assert "MIPROv2" in names
        assert "GEPA" in names

    def test_save_and_load_optimized(self, tmp_path):
        from mosaicx.evaluation.optimize import save_optimized, load_optimized
        import dspy

        # Create a trivial module
        class Dummy(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.Predict("question -> answer")

        module = Dummy()
        path = tmp_path / "optimized.json"
        save_optimized(module, path)
        assert path.exists()

        loaded = load_optimized(Dummy, path)
        assert isinstance(loaded, Dummy)
```

**Step 2–5: Implement, test, commit**

Create `mosaicx/evaluation/optimize.py` with:
- `OPTIMIZATION_STRATEGY` — the 3-tier progressive strategy list
- `get_optimizer_config(budget)` — budget presets
- `save_optimized(module, path)` / `load_optimized(cls, path)` — serialization
- `run_optimization(module, trainset, valset, metric, budget)` — orchestrator

```bash
git commit -m "feat: add optimization workflow with progressive GEPA strategy"
```

---

### Task 19: Batch processor with checkpointing

**Files:**
- Create: `mosaicx/batch.py`
- Create: `tests/test_batch.py`

**Step 1: Write the failing test**

```python
# tests/test_batch.py
"""Tests for the batch processing engine."""

import pytest
from pathlib import Path
import json


class TestBatchCheckpointing:
    def test_checkpoint_save_load(self, tmp_path):
        from mosaicx.batch import BatchCheckpoint

        cp = BatchCheckpoint(batch_id="test_batch", checkpoint_dir=tmp_path)
        cp.mark_completed("doc1.pdf", {"status": "ok"})
        cp.mark_completed("doc2.pdf", {"status": "ok"})
        cp.save()

        cp2 = BatchCheckpoint.load(tmp_path / "test_batch.json")
        assert cp2.is_completed("doc1.pdf")
        assert cp2.is_completed("doc2.pdf")
        assert not cp2.is_completed("doc3.pdf")

    def test_checkpoint_resume_skips_completed(self, tmp_path):
        from mosaicx.batch import BatchCheckpoint

        cp = BatchCheckpoint(batch_id="test", checkpoint_dir=tmp_path)
        cp.mark_completed("doc1.pdf", {"status": "ok"})
        cp.save()

        all_docs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        remaining = [d for d in all_docs if not cp.is_completed(d)]
        assert remaining == ["doc2.pdf", "doc3.pdf"]


class TestBatchProcessor:
    def test_processor_creation(self):
        from mosaicx.batch import BatchProcessor
        proc = BatchProcessor(workers=2)
        assert proc.workers == 2

    def test_error_isolation(self, tmp_path):
        from mosaicx.batch import BatchProcessor

        # Create some dummy files
        for i in range(3):
            (tmp_path / f"doc{i}.txt").write_text(f"Content {i}")

        proc = BatchProcessor(workers=1)
        # Just verify it can be instantiated and has process method
        assert hasattr(proc, "process_directory")
```

**Step 2–5: Implement, test, commit**

```bash
git commit -m "feat: add batch processor with checkpointing and error isolation"
```

---

## Phase 6: Export & Polish

### Task 20: Tabular export (CSV, Parquet, JSONL)

**Files:**
- Create: `mosaicx/export/__init__.py`
- Create: `mosaicx/export/tabular.py`
- Create: `tests/test_export_tabular.py`

**Step 1: Write the failing test**

```python
# tests/test_export_tabular.py
"""Tests for tabular export (CSV, Parquet, JSONL)."""

import pytest
import json
from pathlib import Path


SAMPLE_RESULTS = [
    {
        "source_file": "report1.pdf",
        "exam_type": "chest_ct",
        "indication": "Cough",
        "findings": [
            {"anatomy": "RUL", "observation": "nodule", "size_mm": 5.0},
            {"anatomy": "RLL", "observation": "atelectasis"},
        ],
        "impression": "Pulmonary nodule.",
        "completeness": 0.85,
    },
    {
        "source_file": "report2.pdf",
        "exam_type": "chest_ct",
        "indication": "Follow-up",
        "findings": [{"anatomy": "RUL", "observation": "nodule", "size_mm": 5.0}],
        "impression": "Stable nodule.",
        "completeness": 0.90,
    },
]


class TestCSVExport:
    def test_one_row_strategy(self, tmp_path):
        from mosaicx.export.tabular import export_csv

        path = tmp_path / "output.csv"
        export_csv(SAMPLE_RESULTS, path, strategy="one_row")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_findings_rows_strategy(self, tmp_path):
        from mosaicx.export.tabular import export_csv

        path = tmp_path / "output.csv"
        export_csv(SAMPLE_RESULTS, path, strategy="findings_rows")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows (2 findings from report1 + 1 from report2)


class TestJSONLExport:
    def test_export_jsonl(self, tmp_path):
        from mosaicx.export.tabular import export_jsonl

        path = tmp_path / "output.jsonl"
        export_jsonl(SAMPLE_RESULTS, path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "source_file" in obj


class TestParquetExport:
    def test_export_parquet(self, tmp_path):
        from mosaicx.export.tabular import export_parquet

        path = tmp_path / "output.parquet"
        export_parquet(SAMPLE_RESULTS, path, strategy="one_row")
        assert path.exists()
        assert path.stat().st_size > 0
```

**Step 2–5: Implement, test, commit**

```bash
git commit -m "feat: add tabular export (CSV, Parquet, JSONL) with flattening strategies"
```

---

### Task 21: FHIR bundle export

**Files:**
- Create: `mosaicx/export/fhir_bundle.py`
- Create: `tests/test_export_fhir.py`

**Step 1: Write the failing test**

Test that `export_fhir_bundle()` takes a list of structured results and writes a FHIR Bundle JSON file per result.

**Step 2–5: Implement, test, commit**

```bash
git commit -m "feat: add FHIR R4 bundle file export"
```

---

### Task 22: Narrative report export (PDF/DOCX/Markdown)

**Files:**
- Create: `mosaicx/export/report.py`
- Create: `tests/test_export_report.py`

**Step 1: Write the failing test**

Test Markdown export (no optional deps needed). PDF and DOCX export should gracefully skip if `reportlab`/`python-docx` are not installed.

**Step 2–5: Implement, test, commit**

```bash
git commit -m "feat: add narrative report export (Markdown, optional PDF/DOCX)"
```

---

### Task 23: Wire CLI commands to real pipelines

**Files:**
- Modify: `mosaicx/cli.py` — connect `extract`, `batch`, `schema generate`, `summarize`, `deidentify`, `optimize`, `config show` to actual implementations

**Step 1: Write the failing test**

```python
# tests/test_cli_integration.py
"""Integration tests for CLI commands with mocked LLM."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch


class TestExtractCommand:
    def test_extract_txt_file(self, tmp_path):
        from mosaicx.cli import cli

        # Create a sample text file
        txt = tmp_path / "report.txt"
        txt.write_text("Patient presents with cough. CT shows 5mm nodule in RUL.")

        runner = CliRunner()
        # Just verify the command doesn't crash with a text file
        result = runner.invoke(cli, ["extract", "--document", str(txt)])
        assert result.exit_code == 0


class TestConfigShowCommand:
    def test_config_show(self):
        from mosaicx.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "lm" in result.output
```

**Step 2–5: Wire commands, test, commit**

```bash
git commit -m "feat: wire CLI commands to pipeline implementations"
```

---

### Task 24: Update __init__.py public API for v2

**Files:**
- Modify: `mosaicx/__init__.py`

**Step 1: Write the failing test**

```python
# tests/test_public_api.py
"""Test the v2 public API surface."""

import pytest


class TestPublicAPI:
    def test_extract_function(self):
        from mosaicx import extract
        assert callable(extract)

    def test_summarize_function(self):
        from mosaicx import summarize
        assert callable(summarize)

    def test_generate_schema_function(self):
        from mosaicx import generate_schema
        assert callable(generate_schema)

    def test_deidentify_function(self):
        from mosaicx import deidentify
        assert callable(deidentify)
```

**Step 2–5: Add public API functions as thin wrappers, test, commit**

```bash
git commit -m "feat: expose v2 public API (extract, summarize, generate_schema, deidentify)"
```

---

### Task 25: Final integration test and cleanup

**Files:**
- Create: `tests/integration/test_e2e.py`
- Modify: `mosaicx/__init__.py` — ensure version is correct

**Step 1: Write the integration test**

```python
# tests/integration/test_e2e.py
"""End-to-end integration test — document loading → extraction → export."""

import pytest
from pathlib import Path


@pytest.mark.integration
class TestEndToEnd:
    def test_txt_to_json_extraction(self, tmp_path):
        """Load a text file, extract with generic pipeline (mocked LLM), export JSONL."""
        from mosaicx.documents import load_document

        # Create test document
        report = tmp_path / "report.txt"
        report.write_text(
            "Patient: John Doe, 65M.\n"
            "Indication: Persistent cough.\n"
            "Findings: 5mm ground glass nodule in the right upper lobe.\n"
            "Impression: Pulmonary nodule, recommend follow-up CT in 12 months."
        )

        doc = load_document(report)
        assert not doc.is_empty
        assert "5mm" in doc.text

    def test_template_compilation_roundtrip(self, tmp_path):
        """YAML template → Pydantic model → instantiation."""
        from mosaicx.schemas.template_compiler import compile_template

        yaml_content = """\
name: TestReport
description: Test
sections:
  - name: summary
    type: str
    required: true
"""
        Model = compile_template(yaml_content)
        instance = Model(summary="Normal findings.")
        assert instance.summary == "Normal findings."

    def test_completeness_scoring(self):
        """Completeness evaluator on a Pydantic model."""
        from mosaicx.evaluation.completeness import compute_completeness
        from pydantic import BaseModel

        class MiniReport(BaseModel):
            indication: str
            findings: list[str]

        report = MiniReport(indication="Cough", findings=["nodule"])
        result = compute_completeness(report, source_text="Cough. 5mm nodule in RUL.")
        assert 0.0 <= result["overall"] <= 1.0

    def test_fhir_bundle_structure(self):
        """FHIR bundle has valid structure."""
        from mosaicx.schemas.fhir import build_diagnostic_report
        bundle = build_diagnostic_report(
            patient_id="TEST001",
            findings=[{"anatomy": "RUL", "observation": "nodule"}],
            impression="Nodule.",
        )
        assert bundle["resourceType"] == "Bundle"
```

**Step 2: Run the full test suite**

Run: `uv run pytest tests/ -x -v --tb=short`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/integration/test_e2e.py
git commit -m "test: add end-to-end integration tests for v2 pipeline"
```

---

### Task 26: Remove v1 dead code

**Files:**
- Remove or archive: `mosaicx/extractor.py`, `mosaicx/standardizer.py`, `mosaicx/text_extraction.py`, `mosaicx/prompting/`, `mosaicx/utils/config.py`, `mosaicx/utils/pathing.py`, `mosaicx/api/` (old), `mosaicx/cli/` (old dir), `mosaicx/mosaicx.py`, `mosaicx/schema/builder.py` (old exec-based), `webapp/`
- Modify: `mosaicx/__main__.py` — point to new CLI

**Step 1: Verify all new tests pass**

Run: `uv run pytest tests/ -x -v`

**Step 2: Remove dead modules one at a time, re-run tests after each removal**

Only remove files that are no longer imported anywhere. Use `grep` to verify no imports reference the old modules before deletion.

**Step 3: Commit**

```bash
git commit -m "chore: remove v1 dead code (extractor, standardizer, webapp, etc.)"
```

---

### Task 27: Final verification and version bump

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -v --tb=short`
Expected: All tests PASS

**Step 2: Run linter**

Run: `uv run ruff check mosaicx/`
Expected: Clean or only minor warnings

**Step 3: Verify CLI works**

Run: `uv run mosaicx --version`
Expected: `mosaicx 2.0.0a1`

Run: `uv run mosaicx --help`
Expected: Shows all command groups

**Step 4: Bump version to 2.0.0a1 in pyproject.toml if not already done**

**Step 5: Commit**

```bash
git commit -m "chore: v2.0.0a1 — complete rewrite with DSPy pipelines"
```

---

## Summary

| Phase | Tasks | Key deliverables |
|-------|-------|-----------------|
| **1. Core Infrastructure** | 1-5 | Config, document loader, CLI skeleton, display, package init |
| **2. DSPy Pipelines** | 6-11 | Schema gen, template compiler, extraction, radiology, summarizer, deidentifier |
| **3. RadReport Templates** | 12-13 | Template registry, scoring systems |
| **4. Quality & Ontology** | 14-17 | Completeness evaluator, metrics/rewards, ontology resolver, FHIR |
| **5. Optimization & Batch** | 18-19 | GEPA workflow, batch processor |
| **6. Export & Polish** | 20-27 | Tabular/FHIR/narrative export, CLI wiring, public API, cleanup, integration tests |

**Total: 27 tasks, ~100 TDD steps**

Each task follows RED-GREEN-REFACTOR: write failing test → implement → verify pass → commit.

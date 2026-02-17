# mosaicx/cli.py
"""
MOSAICX CLI v2 -- Click commands with DigiTx-inspired terminal UI.

Provides the ``mosaicx`` console entry-point declared in pyproject.toml as
``mosaicx.cli:cli``.  Commands call into the actual pipeline modules:

- extract:    DocumentExtractor (DSPy) -- single file or batch via --dir
- template:   template management (create, list, show, refine, migrate, etc.)
- summarize:  ReportSummarizer (DSPy)
- deidentify: Deidentifier (DSPy) or regex_scrub_phi (--regex-only)
- optimize:   get_optimizer_config
- config:     MosaicxConfig display
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Optional

import click
from rich import box
from rich.console import Console
from rich.markup import escape as _esc
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

from . import cli_theme as theme
from .config import get_config

console = Console()

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_VERSION_NUMBER = "2.0.0a1"


def _print_version(
    ctx: click.Context,
    _param: click.Parameter,
    value: bool,
) -> None:
    if not value or ctx.resilient_parsing:
        return
    theme.print_version(_VERSION_NUMBER, console)
    ctx.exit()


# ---------------------------------------------------------------------------
# DSPy configuration helper
# ---------------------------------------------------------------------------


def _check_api_key() -> None:
    """Fast preflight check — fail before loading documents."""
    cfg = get_config()
    if not cfg.api_key:
        raise click.ClickException(
            "No API key configured. Set MOSAICX_API_KEY or add api_key to your config."
        )


def _configure_dspy() -> None:
    """Configure DSPy with the LM from MosaicxConfig.

    Raises ``click.ClickException`` if DSPy cannot be imported or the
    API key is missing.
    """
    cfg = get_config()
    if not cfg.api_key:
        raise click.ClickException(
            "No API key configured. Set MOSAICX_API_KEY or add api_key to your config."
        )
    try:
        import dspy
    except ImportError:
        raise click.ClickException(
            "DSPy is required for this command. Install with: pip install dspy"
        )
    from .metrics import TokenTracker, make_harmony_lm, set_tracker

    lm = make_harmony_lm(cfg.lm, api_key=cfg.api_key, api_base=cfg.api_base, temperature=cfg.lm_temperature)
    dspy.configure(lm=lm)

    # Install token usage tracker

    tracker = TokenTracker()
    set_tracker(tracker)
    dspy.settings.usage_tracker = tracker
    dspy.settings.track_usage = True

    model_name = cfg.lm.split("/", 1)[-1] if "/" in cfg.lm else cfg.lm
    console.print(theme.info(f"Model: {model_name}"))


# ---------------------------------------------------------------------------
# Document loading helper (OCR config wiring)
# ---------------------------------------------------------------------------


def _load_doc_with_config(path: Path) -> "LoadedDocument":
    """Load a document using OCR settings from config."""
    from .documents.loader import load_document
    from .documents.models import LoadedDocument  # noqa: F811 — for type only

    cfg = get_config()
    return load_document(
        path,
        ocr_engine=cfg.ocr_engine,
        force_ocr=cfg.force_ocr,
        ocr_langs=cfg.ocr_langs,
        chandra_backend=cfg.chandra_backend if cfg.chandra_backend != "auto" else None,
        quality_threshold=cfg.quality_threshold,
        page_timeout=cfg.ocr_page_timeout,
    )


# ---------------------------------------------------------------------------
# DIGITX-styled Click help
# ---------------------------------------------------------------------------


def _render_styled_help(plain: str, width: int = 80) -> str:
    """Re-render Click help with DIGITX coral/greige colors + 2-space indent."""
    buf = io.StringIO()
    # Extra width avoids Rich re-wrapping lines after we add 2-space indent
    rc = Console(file=buf, force_terminal=True, width=width + 4, highlight=False)
    section: str | None = None

    for line in plain.splitlines():
        stripped = line.strip()
        if not stripped:
            rc.print()
            continue

        # Section headers (no leading whitespace in Click output)
        if line == stripped:
            if stripped.startswith("Usage:"):
                rest = stripped[6:].strip()
                rc.print(
                    f"  [bold {theme.CORAL}]Usage:[/bold {theme.CORAL}]"
                    f" [{theme.GREIGE}]{_esc(rest)}[/{theme.GREIGE}]"
                )
                section = None
                continue

            bare = stripped.rstrip(":")
            if bare in ("Options", "Commands", "Arguments"):
                rc.print(f"  [bold {theme.CORAL}]{stripped}[/bold {theme.CORAL}]")
                section = bare.lower()
                continue

        # Command entries (e.g. "  generate  Description here")
        if section == "commands":
            m = re.match(r"^(\s+)(\S+)(\s{2,})(.+)$", line)
            if m:
                ind, name, gap, desc = m.groups()
                rc.print(
                    f"  {ind}[bold {theme.CORAL}]{_esc(name)}[/bold {theme.CORAL}]"
                    f"{gap}[{theme.MUTED}]{_esc(desc)}[/{theme.MUTED}]"
                )
                continue

        # Option entries (e.g. "  --flag TEXT  Description here")
        if section == "options":
            m = re.match(r"^(\s+)(-.+?)(\s{2,})(.+)$", line)
            if m:
                ind, flags, gap, desc = m.groups()
                rc.print(
                    f"  {ind}[{theme.GREIGE}]{_esc(flags)}[/{theme.GREIGE}]"
                    f"{gap}[{theme.MUTED}]{_esc(desc)}[/{theme.MUTED}]"
                )
                continue

        # Continuation lines (inside a section) or docstring
        if section:
            rc.print(f"  [{theme.MUTED}]{_esc(line)}[/{theme.MUTED}]")
        else:
            rc.print(f"  [{theme.MUTED}]{_esc(stripped)}[/{theme.MUTED}]")

    return buf.getvalue()


class MosaicxGroup(click.Group):
    """Click group with DIGITX-styled help output."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # Show banner at the top of root-level help (mosaicx --help)
        if ctx.parent is None:
            cfg = get_config()
            theme.print_banner(_VERSION_NUMBER, console, lm=cfg.lm, lm_cheap=cfg.lm_cheap)
        tmp = click.HelpFormatter(width=formatter.width)
        super().format_help(ctx, tmp)
        formatter.write(_render_styled_help(tmp.getvalue(), formatter.width or 80))

    def group(self, *args, **kwargs):
        kwargs.setdefault("cls", MosaicxGroup)
        return super().group(*args, **kwargs)

    def command(self, *args, **kwargs):
        kwargs.setdefault("cls", MosaicxCommand)
        return super().command(*args, **kwargs)


class MosaicxCommand(click.Command):
    """Click command with DIGITX-styled help output."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        tmp = click.HelpFormatter(width=formatter.width)
        super().format_help(ctx, tmp)
        formatter.write(_render_styled_help(tmp.getvalue(), formatter.width or 80))


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------


@click.group(invoke_without_command=True, cls=MosaicxGroup)
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show version and exit.",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """MOSAICX -- Medical cOmputational Suite for Advanced Intelligent eXtraction."""
    cfg = get_config()
    if ctx.invoked_subcommand is not None:
        # Subcommand execution — show banner before running the command
        theme.print_banner(_VERSION_NUMBER, console, lm=cfg.lm, lm_cheap=cfg.lm_cheap)
    else:
        # No subcommand — show help (banner included via format_help)
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# extract -- batch helper
# ---------------------------------------------------------------------------


def _extract_batch(
    ctx: click.Context,
    directory: Path,
    template: Optional[str],
    mode: Optional[str],
    optimized: Optional[Path],
    output_dir_path: Optional[Path],
    formats: tuple[str, ...],
    workers: int,
    resume: bool,
) -> None:
    """Batch-process a directory of documents (called from the extract command)."""
    # Preflight: check API key before expensive document loading
    _check_api_key()

    # Validate mode name early (before configuring DSPy)
    if mode is not None and template is None:
        import mosaicx.pipelines.pathology  # noqa: F401
        import mosaicx.pipelines.radiology  # noqa: F401
        from .pipelines.modes import get_mode
        try:
            get_mode(mode)
        except ValueError as exc:
            raise click.ClickException(str(exc))

    from .batch import BatchProcessor

    cfg = get_config()

    # Default output dir to a sibling of the input directory
    if output_dir_path is None:
        output_dir_path = directory.parent / f"{directory.name}_output"

    effective_formats = formats if formats else tuple(cfg.default_export_formats)
    effective_workers = workers

    processor = BatchProcessor(
        workers=effective_workers,
        checkpoint_every=cfg.checkpoint_every,
    )

    theme.section("Batch Processing", console, "01")
    t = theme.make_clean_table(show_header=False)
    t.add_column("Key", style=f"bold {theme.CORAL}", no_wrap=True)
    t.add_column("Value")
    t.add_row("Input directory", str(directory))
    t.add_row("Output directory", str(output_dir_path))
    if template:
        t.add_row("Template", template)
    if mode:
        t.add_row("Mode", mode)
    t.add_row("Export formats", ", ".join(effective_formats))
    t.add_row("Workers", str(effective_workers))
    t.add_row("Resume", theme.badge("Yes", "stable") if resume else "No")
    console.print(Padding(t, (0, 0, 0, 2)))

    # Build process function based on extraction path
    from .metrics import PipelineMetrics
    batch_metrics = PipelineMetrics()

    if template:
        # Resolve template once, then use for all documents
        from .report import detect_mode, resolve_template

        try:
            template_model, tpl_name = resolve_template(template=template)
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc))

        if tpl_name:
            console.print(theme.info(f"Template: {tpl_name}"))

        effective_mode = detect_mode(tpl_name)

        if effective_mode is not None and template_model is None:
            # Mode pipeline (no YAML schema)
            import mosaicx.pipelines.radiology  # noqa: F401
            import mosaicx.pipelines.pathology  # noqa: F401
            from mosaicx.pipelines.extraction import extract_with_mode

            console.print(theme.info(f"Mode: {effective_mode}"))
            _configure_dspy()

            def process_fn(text: str) -> dict:
                output_data, metrics = extract_with_mode(text, effective_mode)
                if metrics is not None:
                    batch_metrics.steps.extend(metrics.steps)
                return output_data
        elif template_model is not None:
            from mosaicx.pipelines.extraction import DocumentExtractor

            _configure_dspy()
            extractor = DocumentExtractor(output_schema=template_model)
            if optimized is not None:
                from .evaluation.optimize import load_optimized
                extractor = load_optimized(type(extractor), optimized)

            def process_fn(text: str) -> dict:
                result = extractor(document_text=text)
                doc_metrics = getattr(extractor, "_last_metrics", None)
                if doc_metrics is not None:
                    batch_metrics.steps.extend(doc_metrics.steps)
                output = {}
                if hasattr(result, "extracted"):
                    val = result.extracted
                    output["extracted"] = val.model_dump() if hasattr(val, "model_dump") else val
                return output
        else:
            raise click.ClickException(
                f"Template {template!r} resolved but produced no extraction template."
            )
    elif mode:
        import mosaicx.pipelines.radiology  # noqa: F401
        import mosaicx.pipelines.pathology  # noqa: F401
        from mosaicx.pipelines.extraction import extract_with_mode
        _configure_dspy()

        def process_fn(text: str) -> dict:
            output_data, metrics = extract_with_mode(text, mode)
            if metrics is not None:
                batch_metrics.steps.extend(metrics.steps)
            return output_data
    else:
        from mosaicx.pipelines.extraction import DocumentExtractor
        _configure_dspy()
        extractor = DocumentExtractor()
        if optimized is not None:
            from .evaluation.optimize import load_optimized
            extractor = load_optimized(type(extractor), optimized)

        def process_fn(text: str) -> dict:
            result = extractor(document_text=text)
            doc_metrics = getattr(extractor, "_last_metrics", None)
            if doc_metrics is not None:
                batch_metrics.steps.extend(doc_metrics.steps)
            output = {}
            if hasattr(result, "extracted"):
                val = result.extracted
                output["extracted"] = val.model_dump() if hasattr(val, "model_dump") else val
            return output

    resume_id = "resume" if resume else None
    checkpoint_dir = output_dir_path / ".checkpoints" if resume else None

    # Count documents for progress bar
    supported = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    total_docs = sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in supported)

    with theme.progress(total_docs, "documents", console) as advance:
        result = processor.process_directory(
            input_dir=directory,
            output_dir=output_dir_path,
            process_fn=process_fn,
            resume_id=resume_id,
            checkpoint_dir=checkpoint_dir,
            load_fn=lambda p: _load_doc_with_config(p).text,
            on_progress=lambda name, success: advance(),
        )

    console.print(theme.ok(f"Batch complete -- {result['succeeded']}/{result['total']} succeeded"))
    if result["skipped"]:
        console.print(theme.info(f"{result['skipped']} skipped (already processed)"))
    if result["failed"]:
        console.print(theme.warn(f"{result['failed']} failed"))
        for err in result.get("errors", []):
            console.print(theme.info(f"{err['file']}: {err['error']}"))

    # -- Export consolidated formats -----------------------------------------
    json_files = sorted(
        p for p in output_dir_path.glob("*.json")
        if not p.name.startswith(".")
    )

    if "jsonl" in effective_formats and json_files:
        jsonl_path = output_dir_path / "results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for jf in json_files:
                data = json.loads(jf.read_text(encoding="utf-8"))
                data["_source"] = jf.stem
                f.write(json.dumps(data, default=str, ensure_ascii=False) + "\n")
        console.print(theme.ok(f"Exported {jsonl_path}"))

    if "csv" in effective_formats and json_files:
        try:
            import pandas as pd

            records = []
            for jf in json_files:
                data = json.loads(jf.read_text(encoding="utf-8"))
                data["_source"] = jf.stem
                records.append(data)
            df = pd.json_normalize(records, sep="_")
            csv_path = output_dir_path / "results.csv"
            df.to_csv(csv_path, index=False)
            console.print(theme.ok(f"Exported {csv_path}"))
        except ImportError:
            console.print(theme.warn(
                "pandas required for CSV export: pip install pandas"
            ))

    if "parquet" in effective_formats and json_files:
        try:
            import pandas as pd

            records = []
            for jf in json_files:
                data = json.loads(jf.read_text(encoding="utf-8"))
                data["_source"] = jf.stem
                records.append(data)
            df = pd.json_normalize(records, sep="_")
            parquet_path = output_dir_path / "results.parquet"
            df.to_parquet(parquet_path, index=False)
            console.print(theme.ok(f"Exported {parquet_path}"))
        except ImportError:
            console.print(theme.warn(
                "pandas + pyarrow required for parquet: pip install pandas pyarrow"
            ))

    # Display aggregate performance metrics
    if batch_metrics.steps:
        from .cli_display import render_metrics
        render_metrics(batch_metrics, console)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--document", type=click.Path(exists=True, path_type=Path), default=None, help="Path to clinical document (single file).")
@click.option("--dir", "directory", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None, help="Directory of documents to process (batch mode).")
@click.option("--template", type=str, default=None, help="Template name, YAML file path, or saved template name.")
@click.option("--mode", type=str, default=None, help="Extraction mode name (e.g., radiology, pathology).")
@click.option("--score", is_flag=True, default=False, help="Score completeness of extracted data against the template.")
@click.option("--optimized", type=click.Path(exists=True, path_type=Path), default=None, help="Path to optimized program.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save output to file (.json or .yaml/.yml) for single-file extraction.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Output directory for batch results (used with --dir).")
@click.option("--format", "formats", type=click.Choice(["json", "jsonl", "csv", "parquet"], case_sensitive=False), multiple=True, default=("json",), show_default=True, help="Output format(s) for batch results.")
@click.option("--workers", type=int, default=1, show_default=True, help="Number of parallel workers for batch processing.")
@click.option("--resume", is_flag=True, default=False, help="Resume batch processing from last checkpoint.")
@click.option("--list-modes", is_flag=True, default=False, help="Print available modes and exit.")
@click.pass_context
def extract(
    ctx: click.Context,
    document: Optional[Path],
    directory: Optional[Path],
    template: Optional[str],
    mode: Optional[str],
    score: bool,
    optimized: Optional[Path],
    output: Optional[Path],
    output_dir: Optional[Path],
    formats: tuple[str, ...],
    workers: int,
    resume: bool,
    list_modes: bool,
) -> None:
    """Extract structured data from a clinical document or directory.

    \b
    Supports single-file extraction (--document) and batch processing
    (--dir). When --dir is used, all supported documents in the directory
    are processed and results are written to --output-dir.

    \b
    The --template flag is the unified way to specify what to extract.
    It resolves in order:
      1. YAML file path (if suffix is .yaml/.yml and file exists)
      2. Built-in template name (e.g. chest_ct, brain_mri)
      3. Legacy saved schema (from ~/.mosaicx/schemas/)

    \b
    Examples (single file):
      mosaicx extract --document scan.pdf
      mosaicx extract --document scan.pdf --template chest_ct
      mosaicx extract --document scan.pdf --template chest_ct --score

    \b
    Examples (batch / directory):
      mosaicx extract --dir reports/ --template chest_ct
      mosaicx extract --dir reports/ --workers 4 --output-dir results/
      mosaicx extract --dir reports/ --format json csv
      mosaicx extract --dir reports/ --resume
    """

    # --list-modes: print available modes and exit (no document needed)
    if list_modes:
        import mosaicx.pipelines.radiology  # noqa: F401
        import mosaicx.pipelines.pathology  # noqa: F401
        from .pipelines.modes import list_modes as _list_modes

        theme.section("Extraction Modes", console)
        t = theme.make_clean_table(width=len(theme.TAGLINE))
        t.add_column("Mode", style=f"bold {theme.CORAL}", no_wrap=True)
        t.add_column("Description", style=theme.MUTED, ratio=1)
        for name, desc in _list_modes():
            t.add_row(name, desc)
        console.print(Padding(t, (0, 0, 0, 2)))
        console.print()
        console.print(theme.info(f"{len(_list_modes())} mode(s) available"))
        console.print(theme.info("Use --template <name> for template-based extraction"))
        ctx.exit()
        return

    # Mutual exclusivity: --document and --dir
    if document is not None and directory is not None:
        raise click.UsageError(
            "--document and --dir are mutually exclusive. Provide one or the other."
        )

    # Validate: either --document or --dir is required for extraction
    if document is None and directory is None:
        raise click.ClickException(
            "--document or --dir is required. Usage:\n"
            "  mosaicx extract --document <file>    (single file)\n"
            "  mosaicx extract --dir <directory>     (batch mode)"
        )

    # Mutual exclusivity: --template and --mode
    if template is not None and mode is not None:
        raise click.ClickException(
            "--template and --mode are mutually exclusive. Provide at most one."
        )

    # Route to batch processing when --dir is used
    if directory is not None:
        _extract_batch(
            ctx=ctx,
            directory=directory,
            template=template,
            mode=mode,
            optimized=optimized,
            output_dir_path=output_dir,
            formats=formats,
            workers=workers,
            resume=resume,
        )
        return

    # Validate template/mode early (cheap checks before expensive I/O)
    if template is not None:
        from .report import resolve_template
        try:
            resolve_template(template=template)
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc))

    if mode is not None and template is None:
        import mosaicx.pipelines.pathology  # noqa: F401
        import mosaicx.pipelines.radiology  # noqa: F401
        from .pipelines.modes import get_mode
        try:
            get_mode(mode)
        except ValueError as exc:
            raise click.ClickException(str(exc))

    # Preflight: check API key before expensive document loading
    _check_api_key()

    # Load the document
    from .documents.models import DocumentLoadError

    try:
        doc = _load_doc_with_config(document)
    except (FileNotFoundError, ValueError, DocumentLoadError) as exc:
        raise click.ClickException(str(exc))

    if doc.quality_warning:
        console.print(theme.warn("Low OCR quality detected \u2014 results may be unreliable"))

    if doc.is_empty:
        hint = " Try --force-ocr if the document is a scanned image." if document.suffix.lower() == ".pdf" else ""
        raise click.ClickException(f"Document is empty: {document}.{hint}")

    # Build a descriptive load line
    parts = [f"{doc.format} document", f"{doc.char_count:,} chars"]
    if doc.ocr_engine_used:
        parts.append(f"OCR: {doc.ocr_engine_used}")
    if doc.ocr_confidence:
        parts.append(f"confidence: {doc.ocr_confidence:.0%}")
    console.print(theme.info(" \u00b7 ".join(parts)))

    # Determine extraction path
    output_data: dict = {}
    metrics = None  # PipelineMetrics, if available
    _extract_model_instance = None  # Pydantic model for completeness scoring

    if template is not None:
        # Unified template resolution
        from .report import detect_mode, resolve_template

        try:
            template_model, tpl_name = resolve_template(template=template)
        except (ValueError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc))

        if tpl_name:
            console.print(theme.info(f"Template: {tpl_name}"))

        # Detect mode from template (for built-in templates)
        effective_mode = detect_mode(tpl_name)

        if effective_mode is not None and template_model is None:
            # Built-in template with mode pipeline but no YAML schema
            # (e.g. chest_ct without a .yaml file) -- use the mode pipeline
            import mosaicx.pipelines.radiology  # noqa: F401
            import mosaicx.pipelines.pathology  # noqa: F401

            console.print(theme.info(f"Mode: {effective_mode}"))
            _configure_dspy()

            if score:
                from .pipelines.extraction import extract_with_mode_raw
                from .report import _find_primary_model

                with theme.spinner("Extracting with mode... patience you must have", console):
                    output_data, metrics, raw_pred = extract_with_mode_raw(
                        doc.text, effective_mode
                    )
                _extract_model_instance = _find_primary_model(raw_pred)
            else:
                from .pipelines.extraction import extract_with_mode

                with theme.spinner("Extracting with mode... patience you must have", console):
                    output_data, metrics = extract_with_mode(doc.text, effective_mode)
        elif template_model is not None:
            # Template resolved to a Pydantic model -- use DocumentExtractor
            from .pipelines.extraction import DocumentExtractor

            _configure_dspy()
            extractor = DocumentExtractor(output_schema=template_model)
            if optimized is not None:
                from .evaluation.optimize import load_optimized
                extractor = load_optimized(type(extractor), optimized)
            with theme.spinner("Extracting... patience you must have", console):
                result = extractor(document_text=doc.text)
            output_data = {}
            if hasattr(result, "extracted"):
                val = result.extracted
                if hasattr(val, "model_dump"):
                    _extract_model_instance = val
                    output_data["extracted"] = val.model_dump()
                else:
                    output_data["extracted"] = val
            metrics = getattr(extractor, "_last_metrics", None)
        else:
            # Template resolved but no model and no mode -- shouldn't happen
            raise click.ClickException(
                f"Template {template!r} resolved but produced no extraction template."
            )

    elif mode is not None:
        # --mode X: run registered mode pipeline
        import mosaicx.pipelines.radiology  # noqa: F401
        import mosaicx.pipelines.pathology  # noqa: F401
        from .pipelines.modes import get_mode

        # Validate mode name early (before configuring DSPy)
        try:
            get_mode(mode)
        except ValueError as exc:
            raise click.ClickException(str(exc))

        _configure_dspy()
        if score:
            from .pipelines.extraction import extract_with_mode_raw
            from .report import _find_primary_model

            with theme.spinner("Extracting with mode... patience you must have", console):
                output_data, metrics, raw_pred = extract_with_mode_raw(doc.text, mode)
            _extract_model_instance = _find_primary_model(raw_pred)
        else:
            from .pipelines.extraction import extract_with_mode

            with theme.spinner("Extracting with mode... patience you must have", console):
                output_data, metrics = extract_with_mode(doc.text, mode)

    else:
        # Auto mode: no flags -- LLM infers schema
        if score:
            console.print(theme.warn(
                "--score requires a --template or --mode to score against (ignored in auto mode)"
            ))

        from .pipelines.extraction import DocumentExtractor

        _configure_dspy()
        extractor = DocumentExtractor()
        if optimized is not None:
            from .evaluation.optimize import load_optimized
            extractor = load_optimized(type(extractor), optimized)
        with theme.spinner("Extracting... patience you must have", console):
            result = extractor(document_text=doc.text)
        output_data = {}
        if hasattr(result, "extracted"):
            val = result.extracted
            if hasattr(val, "model_dump"):
                _extract_model_instance = val
                output_data["extracted"] = val.model_dump()
            else:
                output_data["extracted"] = val
        if hasattr(result, "inferred_schema"):
            output_data["inferred_schema"] = result.inferred_schema.model_dump()
        metrics = getattr(extractor, "_last_metrics", None)

    console.print(theme.ok("Extracted \u2014 this is the way"))

    # Save to file if --output specified
    if output is not None:
        suffix = output.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise click.ClickException("PyYAML required for YAML output: pip install pyyaml")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(yaml.dump(output_data, default_flow_style=False, sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            # Default to JSON
            if not suffix:
                output = output.with_suffix(".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(output_data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        console.print(theme.ok(f"Saved to {output}"))

    theme.section("Extracted Data", console)
    from .cli_display import render_completeness, render_extracted_data, render_metrics

    render_extracted_data(output_data, console)

    if metrics is not None:
        render_metrics(metrics, console)

    # Completeness scoring (when --score flag is set)
    if score and _extract_model_instance is not None:
        from dataclasses import asdict

        from .evaluation.completeness import compute_report_completeness

        comp = compute_report_completeness(
            _extract_model_instance, doc.text, type(_extract_model_instance)
        )
        render_completeness(asdict(comp), console)

    if output is None:
        console.print()
        console.print(theme.info("Use -o/--output file.json to save full structured data"))


# ---------------------------------------------------------------------------
# pipeline (group)
# ---------------------------------------------------------------------------


@cli.group()
def pipeline() -> None:
    """Scaffold and manage extraction pipelines."""


@pipeline.command("new")
@click.argument("name")
@click.option(
    "--description", "-d",
    type=str,
    default="",
    help="One-line description of the pipeline.",
)
def pipeline_new(name: str, description: str) -> None:
    """Scaffold a new extraction pipeline module.

    NAME is the pipeline name (e.g. cardiology, ophthalmology).
    A snake_case file and PascalCase DSPy Module will be generated
    automatically.

    \b
    Examples:
      mosaicx pipeline new cardiology
      mosaicx pipeline new ophthalmology -d "Ophthalmic imaging reports"
    """
    from .pipelines.scaffold import scaffold_pipeline, WIRING_CHECKLIST, _to_snake_case

    snake = _to_snake_case(name)

    try:
        created = scaffold_pipeline(name, description)
    except FileExistsError as exc:
        raise click.ClickException(str(exc))
    except ValueError as exc:
        raise click.ClickException(str(exc))

    console.print(theme.ok(f"Pipeline scaffolded: {created}"))

    theme.section("Next Steps", console)
    console.print()
    console.print(theme.info(
        "The generated pipeline is runnable but needs to be wired"
    ))
    console.print(theme.info(
        "into the rest of MOSAICX.  Complete these manual steps:"
    ))
    console.print()

    for line in WIRING_CHECKLIST:
        formatted = line.format(name=snake, class_name=_to_snake_case(name).replace("_", " ").title().replace(" ", ""))
        if line.startswith("    "):
            console.print(f"      [{theme.GREIGE}]{formatted.strip()}[/{theme.GREIGE}]")
        elif line == "":
            console.print()
        else:
            console.print(f"  [{theme.CORAL}]>[/{theme.CORAL}] {formatted}")

    console.print()


# ---------------------------------------------------------------------------
# template (group)
# ---------------------------------------------------------------------------


@cli.group()
def template() -> None:
    """Manage extraction templates."""


@template.command("list")
def template_list() -> None:
    """List available built-in and user-created templates.

    \b
    Examples:
      mosaicx template list
    """
    from .schemas.radreport.registry import list_templates

    templates = list_templates()

    theme.section("Built-in Templates", console)

    t = theme.make_clean_table(width=len(theme.TAGLINE))
    t.add_column("Name", style=f"bold {theme.CORAL}", no_wrap=True)
    t.add_column("Mode", style=theme.GREIGE)
    t.add_column("RDES", style="dim")
    t.add_column("Description", style=theme.MUTED, ratio=1)

    for tpl in templates:
        t.add_row(
            tpl.name,
            tpl.mode or "\u2014",
            tpl.radreport_id or "\u2014",
            tpl.description,
        )

    console.print(Padding(t, (0, 0, 0, 2)))
    console.print()
    console.print(theme.info(f"{len(templates)} built-in template(s)"))

    # User templates
    cfg = get_config()
    user_templates: list[tuple[str, str]] = []  # (name, description)
    if cfg.templates_dir.is_dir():
        from .schemas.template_compiler import parse_template

        for f in sorted(cfg.templates_dir.glob("*.yaml")) + sorted(cfg.templates_dir.glob("*.yml")):
            try:
                meta = parse_template(f.read_text(encoding="utf-8"))
                user_templates.append((f.stem, meta.description or "\u2014"))
            except Exception:
                user_templates.append((f.stem, "(invalid YAML)"))

    if user_templates:
        theme.section("User Templates", console)
        ut = theme.make_clean_table(width=len(theme.TAGLINE))
        ut.add_column("Name", style=f"bold {theme.CORAL}", no_wrap=True)
        ut.add_column("Description", style=theme.MUTED, ratio=1)
        for name, desc in user_templates:
            ut.add_row(name, desc)
        console.print(Padding(ut, (0, 0, 0, 2)))
        console.print()
        console.print(theme.info(
            f"{len(user_templates)} user template(s) in {cfg.templates_dir}"
        ))
    else:
        console.print()
        console.print(theme.info(
            f"No user templates yet. Create one with: mosaicx template create --describe \"...\""
        ))


@template.command("validate")
@click.option("--file", "file_path", type=click.Path(exists=True, path_type=Path), required=True, help="Template YAML file to validate.")
def template_validate(file_path: Path) -> None:
    """Validate a template YAML file and show its compiled fields.

    \b
    Examples:
      mosaicx template validate --file my_template.yaml
    """
    from .schemas.template_compiler import compile_template_file

    try:
        model = compile_template_file(file_path)
        console.print(theme.ok("Template is valid \u2014 you shall pass"))
        console.print(theme.info(f"Model: {model.__name__}"))
        console.print(theme.info(f"Fields: {', '.join(model.model_fields.keys())}"))
    except Exception as exc:
        raise click.ClickException(f"Template validation failed: {exc}")


@template.command("create")
@click.option("--describe", type=str, default=None, help="Natural-language description of the template.")
@click.option("--from-document", "from_document", type=click.Path(exists=True, path_type=Path), default=None, help="Infer template from a sample document.")
@click.option("--from-url", "from_url", type=str, default=None, help="Infer template from a web page (e.g. RadReport URL).")
@click.option("--from-radreport", "from_radreport", type=str, default=None, help="RadReport template ID (e.g. RPT50890 or 50890).")
@click.option("--from-json", "from_json", type=click.Path(exists=True, path_type=Path), default=None, help="Convert a legacy JSON schema to YAML template.")
@click.option("--name", type=str, default=None, help="Template name (default: LLM-chosen).")
@click.option("--mode", type=str, default=None, help="Pipeline mode to embed (e.g. radiology, pathology).")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Save to this path instead of ~/.mosaicx/templates/.")
def template_create(
    describe: Optional[str],
    from_document: Optional[Path],
    from_url: Optional[str],
    from_radreport: Optional[str],
    from_json: Optional[Path],
    name: Optional[str],
    mode: Optional[str],
    output: Optional[Path],
) -> None:
    """Create a new YAML template from a description, document, URL, or JSON schema.

    \b
    Provide exactly one source. The LLM generates a template and saves
    it to ~/.mosaicx/templates/ (or --output path).

    \b
    Examples:
      mosaicx template create --describe "chest CT with nodules and lung-rads"
      mosaicx template create --from-document sample_report.pdf
      mosaicx template create --from-radreport 50890
      mosaicx template create --from-json old_schema.json
      mosaicx template create --from-url https://radreport.org/home/50
    """
    from .schemas.template_compiler import schema_spec_to_template_yaml

    sources = sum(x is not None for x in (describe, from_document, from_url, from_radreport, from_json))
    if sources == 0:
        raise click.ClickException(
            "Provide --describe, --from-document, --from-url, --from-radreport, or --from-json."
        )
    if sources > 1 and from_json is not None:
        raise click.ClickException("--from-json cannot be combined with other sources.")

    # --- Path 1: Convert existing JSON schema to YAML ---
    if from_json is not None:
        from .pipelines.schema_gen import SchemaSpec

        try:
            spec = SchemaSpec.model_validate_json(
                from_json.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise click.ClickException(f"Invalid JSON schema: {exc}")

        if name:
            spec.class_name = name

        yaml_str = schema_spec_to_template_yaml(spec, mode=mode)
        dest = _save_template_yaml(spec.class_name, yaml_str, output)
        console.print(theme.ok("Template created from JSON schema"))
        console.print(theme.info(f"Name: {spec.class_name}"))
        console.print(theme.info(f"Fields: {len(spec.fields)}"))
        console.print(theme.info(f"Saved: {dest}"))
        return

    # --- Path 2: RadReport API (deterministic fetch + LLM enrichment) ---
    if from_radreport is not None:
        try:
            import httpx  # noqa: F811
        except ImportError:
            raise click.ClickException(
                "httpx is required for --from-radreport. Install with: pip install httpx"
            )

        from .schemas.radreport.fetcher import fetch_radreport

        console.print(theme.info(f"Fetching RadReport template {from_radreport}"))
        try:
            rr_template = fetch_radreport(from_radreport)
        except httpx.HTTPError as exc:
            raise click.ClickException(f"Failed to fetch RadReport template: {exc}")
        except ValueError as exc:
            raise click.ClickException(str(exc))

        console.print(theme.ok(f"Fetched: {rr_template.title}"))
        if rr_template.specialty:
            console.print(theme.info(f"Specialty: {', '.join(rr_template.specialty)}"))
        console.print(theme.info(f"Sections: {len(rr_template.sections)}"))

        # Build LLM context from parsed sections
        llm_context = rr_template.to_llm_context()
        rr_description = (
            f"Create a structured extraction template for: {rr_template.title}. "
            f"Infer rich types from the section content: use 'list[object]' for "
            f"repeating structured data (e.g. level-by-level findings), 'enum' for "
            f"categorical fields with fixed options (e.g. severity grades), 'float' "
            f"for measurements, and 'str' for free text. Preserve all section names."
        )
        if describe:
            rr_description = f"{rr_description} Additional instructions: {describe}"

        effective_mode = mode or "radiology"
        if mode is None:
            console.print(theme.info("Mode defaulting to 'radiology' (RadReport source). Use --mode to override."))

        _check_api_key()
        _configure_dspy()

        from .pipelines.schema_gen import SchemaGenerator

        generator = SchemaGenerator()
        with theme.spinner("Enriching template with LLM... hold my beer", console):
            result = generator(
                description=rr_description,
                example_text="",
                document_text=llm_context,
            )

        spec = result.schema_spec
        if name:
            spec.class_name = name
        elif not spec.class_name or spec.class_name == "Schema":
            # Generate a sensible name from the RadReport title
            spec.class_name = re.sub(r"[^a-zA-Z0-9]", "", rr_template.title.title())

        yaml_str = schema_spec_to_template_yaml(spec, mode=effective_mode)
        dest = _save_template_yaml(spec.class_name, yaml_str, output)

        console.print(theme.ok("Template created from RadReport"))
        console.print(theme.info(f"Name: {spec.class_name}"))
        console.print(theme.info(f"Fields: {len(spec.fields)}"))
        console.print(theme.info(f"Saved: {dest}"))

        # Preview the YAML
        theme.section("Template Preview", console)
        from rich.syntax import Syntax

        console.print(Padding(
            Syntax(yaml_str, "yaml", theme="ansi_dark", line_numbers=False),
            (0, 0, 0, 2),
        ))

        metrics = getattr(generator, "_last_metrics", None)
        if metrics is not None:
            from .cli_display import render_metrics

            render_metrics(metrics, console)
        return

    # --- Path 3: LLM-powered generation (--describe / --from-document / --from-url) ---
    _check_api_key()

    description = describe or ""
    document_text = ""
    example_text = ""

    if from_document is not None:
        from .documents.models import DocumentLoadError

        try:
            doc = _load_doc_with_config(from_document)
        except (FileNotFoundError, ValueError, DocumentLoadError) as exc:
            raise click.ClickException(str(exc))
        if doc.quality_warning:
            console.print(theme.warn("Low OCR quality detected \u2014 results may be unreliable"))
        if doc.is_empty:
            raise click.ClickException(f"Document is empty: {from_document}")
        parts = [f"{doc.format} document", f"{doc.char_count:,} chars"]
        if doc.ocr_engine_used:
            parts.append(f"OCR: {doc.ocr_engine_used}")
        console.print(theme.info(" \u00b7 ".join(parts)))
        document_text = doc.text

    if from_url is not None:
        try:
            import httpx
        except ImportError:
            raise click.ClickException(
                "httpx is required for --from-url. Install with: pip install httpx"
            )
        console.print(theme.info(f"Fetching {from_url}"))
        try:
            resp = httpx.get(from_url, follow_redirects=True, timeout=30)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise click.ClickException(f"Failed to fetch URL: {exc}")

        # Extract text from HTML
        page_text = resp.text
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(page_text, "html.parser")
            # Remove script and style elements
            for tag in soup(["script", "style"]):
                tag.decompose()
            page_text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            console.print(theme.warn(
                "beautifulsoup4 not installed -- passing raw HTML to LLM. "
                "Install with: pip install beautifulsoup4"
            ))

        # Use page content as document context for the LLM
        if not description:
            description = f"Infer a structured template from this web page content"
        document_text = page_text

    if not description and not document_text:
        raise click.ClickException(
            "Provide --describe or --from-document (or both)."
        )

    _configure_dspy()

    from .pipelines.schema_gen import SchemaGenerator

    generator = SchemaGenerator()
    with theme.spinner("Generating template... hold my beer", console):
        result = generator(
            description=description,
            example_text=example_text,
            document_text=document_text,
        )

    spec = result.schema_spec
    if name:
        spec.class_name = name

    yaml_str = schema_spec_to_template_yaml(spec, mode=mode)
    dest = _save_template_yaml(spec.class_name, yaml_str, output)

    console.print(theme.ok("Template created"))
    console.print(theme.info(f"Name: {spec.class_name}"))
    console.print(theme.info(f"Fields: {len(spec.fields)}"))
    console.print(theme.info(f"Saved: {dest}"))

    # Preview the YAML
    theme.section("Template Preview", console)
    from rich.syntax import Syntax

    console.print(Padding(
        Syntax(yaml_str, "yaml", theme="ansi_dark", line_numbers=False),
        (0, 0, 0, 2),
    ))

    metrics = getattr(generator, "_last_metrics", None)
    if metrics is not None:
        from .cli_display import render_metrics

        render_metrics(metrics, console)


def _save_template_yaml(
    template_name: str, yaml_str: str, output: Optional[Path]
) -> Path:
    """Save a YAML template string to the templates directory or a custom path."""
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml_str, encoding="utf-8")
        return output

    cfg = get_config()
    cfg.templates_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.templates_dir / f"{template_name}.yaml"
    dest.write_text(yaml_str, encoding="utf-8")
    return dest


@template.command("show")
@click.argument("name")
def template_show(name: str) -> None:
    """Show details of a template (built-in or user-created).

    NAME is the template name (e.g. chest_ct, MedicalReport).
    Use 'mosaicx template list' to see available names.

    \b
    Examples:
      mosaicx template show chest_ct
      mosaicx template show MedicalReport
    """
    from .report import _find_builtin_template_yaml, _find_user_template_yaml
    from .schemas.template_compiler import parse_template

    # Try user template first, then built-in
    path = _find_user_template_yaml(name)
    source = "user"
    if path is None:
        path = _find_builtin_template_yaml(name)
        source = "built-in"

    if path is None:
        # Try as a saved schema
        cfg = get_config()
        schema_path = cfg.schema_dir / f"{name}.json"
        if schema_path.exists():
            # Show as JSON schema instead
            from .pipelines.schema_gen import load_schema

            spec = load_schema(name, cfg.schema_dir)
            theme.section(f"{spec.class_name} (legacy schema)", console)
            if spec.description:
                console.print(Padding(
                    f"[{theme.MUTED}]{spec.description}[/{theme.MUTED}]",
                    (0, 0, 0, 2),
                ))
                console.print()
            t = theme.make_clean_table()
            t.add_column("Field", style=f"bold {theme.CORAL}", no_wrap=True)
            t.add_column("Type", style=theme.GREIGE)
            t.add_column("Req", justify="center", no_wrap=True)
            t.add_column("Description", style=theme.MUTED)
            for f in spec.fields:
                type_label = f.type
                if f.enum_values:
                    type_label = f"enum({', '.join(f.enum_values)})"
                t.add_row(
                    f.name,
                    type_label,
                    "[green]\u2713[/green]" if f.required else "[dim]\u2014[/dim]",
                    f.description or "\u2014",
                )
            console.print(Padding(t, (0, 0, 0, 2)))
            console.print()
            console.print(theme.info(f"{len(spec.fields)} fields"))
            console.print(theme.info(
                "Tip: Convert to YAML with: "
                f"mosaicx template create --from-json {schema_path}"
            ))
            return

        raise click.ClickException(
            f"Template {name!r} not found. "
            f"Use 'mosaicx template list' to see available templates."
        )

    yaml_content = path.read_text(encoding="utf-8")
    meta = parse_template(yaml_content)

    theme.section(f"{meta.name} ({source})", console)

    if meta.description:
        console.print(Padding(
            f"[{theme.MUTED}]{meta.description}[/{theme.MUTED}]",
            (0, 0, 0, 2),
        ))
        console.print()

    # Metadata row
    meta_parts = []
    if meta.mode:
        meta_parts.append(f"Mode: {meta.mode}")
    if meta.radreport_id:
        meta_parts.append(f"RDES: {meta.radreport_id}")
    meta_parts.append(f"Source: {path}")
    for part in meta_parts:
        console.print(theme.info(part))
    console.print()

    # Sections table
    t = theme.make_clean_table()
    t.add_column("Section", style=f"bold {theme.CORAL}", no_wrap=True)
    t.add_column("Type", style=theme.GREIGE)
    t.add_column("Req", justify="center", no_wrap=True)
    t.add_column("Description", style=theme.MUTED, ratio=1)

    for s in meta.sections:
        type_label = s.type
        if s.type == "enum" and s.values:
            type_label = f"enum({', '.join(s.values)})"
        elif s.type == "list" and s.item:
            type_label = f"list[{s.item.type}]"
        t.add_row(
            s.name,
            type_label,
            "[green]\u2713[/green]" if s.required else "[dim]\u2014[/dim]",
            s.description or "\u2014",
        )

    console.print(Padding(t, (0, 0, 0, 2)))
    console.print()
    console.print(theme.info(f"{len(meta.sections)} sections"))


@template.command("refine")
@click.argument("name")
@click.option("--instruction", type=str, required=True, help="Natural-language refinement instruction.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Save refined template to a different path.")
def template_refine(name: str, instruction: str, output: Optional[Path]) -> None:
    """Refine an existing template using LLM-powered instructions.

    NAME is the template name (e.g. chest_ct, MedicalReport).
    Use 'mosaicx template list' to see available names.

    \b
    Examples:
      mosaicx template refine MedicalReport --instruction "add pathology"
      mosaicx template refine chest_ct --instruction "add lung-rads scoring"
    """
    from .report import _find_builtin_template_yaml, _find_user_template_yaml
    from .schemas.template_compiler import parse_template, schema_spec_to_template_yaml

    # Locate the template
    path = _find_user_template_yaml(name)
    if path is None:
        path = _find_builtin_template_yaml(name)

    if path is None:
        raise click.ClickException(
            f"Template {name!r} not found as a YAML template. "
            f"Use 'mosaicx template list' to see available templates."
        )

    yaml_content = path.read_text(encoding="utf-8")
    meta = parse_template(yaml_content)

    # Convert to SchemaSpec for the refiner
    from .pipelines.schema_gen import FieldSpec as GenFieldSpec
    from .pipelines.schema_gen import SchemaSpec

    gen_fields = []
    for s in meta.sections:
        type_str = s.type
        if s.type == "list" and s.item:
            type_str = f"list[{s.item.type}]"
        gen_fields.append(GenFieldSpec(
            name=s.name,
            type=type_str,
            description=s.description or "",
            required=s.required,
            enum_values=list(s.values) if s.values else None,
        ))

    spec = SchemaSpec(
        class_name=meta.name,
        description=meta.description or "",
        fields=gen_fields,
    )

    _configure_dspy()

    from .pipelines.schema_gen import SchemaRefiner

    refiner = SchemaRefiner()
    with theme.spinner("Refining template... one does not simply edit a template", console):
        result = refiner(
            current_schema=spec.model_dump_json(indent=2),
            instruction=instruction,
        )

    refined_spec = result.schema_spec
    yaml_str = schema_spec_to_template_yaml(refined_spec, mode=meta.mode)

    # Archive current version before saving (if user template exists)
    from .report import _find_user_template_yaml as _find_user_tpl
    user_path = _find_user_tpl(name)
    if user_path is not None and output is None:
        _archive_template(name, user_path)

    # Save: user templates always go to user dir (even if refining built-in)
    dest = _save_template_yaml(refined_spec.class_name, yaml_str, output)

    console.print(theme.ok("Template refined"))
    console.print(theme.info(f"Name: {refined_spec.class_name}"))
    console.print(theme.info(f"Fields: {len(refined_spec.fields)}"))
    console.print(theme.info(f"Saved: {dest}"))

    # Show diff
    old_names = {s.name for s in meta.sections}
    new_names = {f.name for f in refined_spec.fields}
    added = new_names - old_names
    removed = old_names - new_names
    if added:
        for n in sorted(added):
            console.print(theme.info(f"+ {n}"))
    if removed:
        for n in sorted(removed):
            console.print(theme.info(f"- {n}"))

    metrics = getattr(refiner, "_last_metrics", None)
    if metrics is not None:
        from .cli_display import render_metrics

        render_metrics(metrics, console)


@template.command("migrate")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be migrated without writing files.")
def template_migrate(dry_run: bool) -> None:
    """Migrate legacy JSON schemas to YAML templates.

    Converts ~/.mosaicx/schemas/*.json to ~/.mosaicx/templates/*.yaml.

    \b
    Examples:
      mosaicx template migrate --dry-run
      mosaicx template migrate
    """
    from .pipelines.schema_gen import load_schema
    from .schemas.template_compiler import schema_spec_to_template_yaml

    cfg = get_config()
    schema_dir = cfg.schema_dir
    templates_dir = cfg.templates_dir

    if not schema_dir.is_dir():
        console.print(theme.info("No schemas directory found. Nothing to migrate."))
        return

    json_files = sorted(schema_dir.glob("*.json"))
    if not json_files:
        console.print(theme.info("No JSON schemas found. Nothing to migrate."))
        return

    if not dry_run:
        templates_dir.mkdir(parents=True, exist_ok=True)

    theme.section("Template Migration", console)

    migrated = 0
    skipped = 0
    errors = 0

    for json_path in json_files:
        name = json_path.stem
        yaml_path = templates_dir / f"{name}.yaml"

        # Skip if YAML already exists
        if yaml_path.exists():
            console.print(theme.info(f"  skip  {name} (YAML already exists)"))
            skipped += 1
            continue

        try:
            spec = load_schema(name, schema_dir)
            yaml_text = schema_spec_to_template_yaml(spec)

            if dry_run:
                console.print(theme.info(f"  would migrate  {name}.json -> {name}.yaml"))
            else:
                yaml_path.write_text(yaml_text, encoding="utf-8")
                console.print(theme.ok(f"  migrated  {name}.json -> {name}.yaml"))
            migrated += 1
        except Exception as exc:
            console.print(theme.warn(f"  error  {name}: {exc}"))
            errors += 1

    console.print()
    parts = []
    if migrated:
        parts.append(f"{migrated} migrated" if not dry_run else f"{migrated} would migrate")
    if skipped:
        parts.append(f"{skipped} skipped")
    if errors:
        parts.append(f"{errors} errors")
    if not parts:
        parts.append("nothing to do")
    console.print(theme.info(", ".join(parts)))

    if dry_run and migrated:
        console.print(theme.info("Run without --dry-run to perform the migration."))
    elif migrated and not dry_run:
        console.print(theme.info("Use 'mosaicx template list' to see all templates"))


# ---------------------------------------------------------------------------
# Template versioning helpers
# ---------------------------------------------------------------------------


def _template_history_dir() -> Path:
    """Return the template history directory, creating it if needed."""
    cfg = get_config()
    hdir = cfg.templates_dir / ".history"
    hdir.mkdir(parents=True, exist_ok=True)
    return hdir


def _list_template_versions(name: str) -> list[tuple[int, Path]]:
    """List archived versions of a user template, sorted by version number."""
    hdir = get_config().templates_dir / ".history"
    if not hdir.is_dir():
        return []
    results: list[tuple[int, Path]] = []
    for p in sorted(hdir.glob(f"{name}_v*.yaml")):
        m = re.match(rf"^{re.escape(name)}_v(\d+)\.yaml$", p.name)
        if m:
            results.append((int(m.group(1)), p))
    return results


def _archive_template(name: str, current_path: Path) -> int:
    """Copy the current template YAML into .history/ and return the version number."""
    hdir = _template_history_dir()
    versions = _list_template_versions(name)
    next_version = (versions[-1][0] + 1) if versions else 1
    import shutil

    dest = hdir / f"{name}_v{next_version}.yaml"
    shutil.copy2(current_path, dest)
    return next_version


@template.command("history")
@click.argument("name")
def template_history(name: str) -> None:
    """Show version history of a user template.

    NAME is the template name (e.g. MedicalReport).

    \b
    Examples:
      mosaicx template history MedicalReport
    """
    from .report import _find_user_template_yaml

    path = _find_user_template_yaml(name)
    if path is None:
        raise click.ClickException(
            f"User template {name!r} not found in {get_config().templates_dir}. "
            f"Only user templates have version history."
        )

    versions = _list_template_versions(name)

    theme.section(f"{name} History", console)

    if not versions:
        console.print(theme.info("No version history yet (only current version exists)"))
        console.print(theme.info("History is created when you use 'template refine' or 'template revert'"))
        return

    from datetime import datetime

    t = theme.make_clean_table(width=len(theme.TAGLINE))
    t.add_column("Version", style=f"bold {theme.CORAL}", no_wrap=True)
    t.add_column("Date", style=theme.MUTED, ratio=1)

    for ver_num, ver_path in versions:
        mtime = datetime.fromtimestamp(ver_path.stat().st_mtime)
        t.add_row(f"v{ver_num}", mtime.strftime("%Y-%m-%d %H:%M"))

    # Current
    from datetime import datetime as _dt

    current_mtime = _dt.fromtimestamp(path.stat().st_mtime)
    t.add_row("current", current_mtime.strftime("%Y-%m-%d %H:%M"))

    console.print(Padding(t, (0, 0, 0, 2)))
    console.print()
    console.print(theme.info(f"{len(versions)} archived version(s) + current"))


@template.command("revert")
@click.argument("name")
@click.option("--version", "version_num", type=int, required=True, help="Version number to revert to.")
def template_revert(name: str, version_num: int) -> None:
    """Revert a user template to a previous version.

    NAME is the template name (e.g. MedicalReport).
    Use 'mosaicx template history NAME' to see available versions.

    \b
    Examples:
      mosaicx template revert MedicalReport --version 2
    """
    import shutil

    from .report import _find_user_template_yaml

    path = _find_user_template_yaml(name)
    if path is None:
        raise click.ClickException(
            f"User template {name!r} not found."
        )

    versions = _list_template_versions(name)
    version_paths = {v: p for v, p in versions}

    if version_num not in version_paths:
        available = ", ".join(f"v{v}" for v, _ in versions) or "none"
        raise click.ClickException(
            f"Version {version_num} not found for {name!r}. Available: {available}"
        )

    # Archive current before reverting
    archive_ver = _archive_template(name, path)

    # Copy target version to current
    shutil.copy2(version_paths[version_num], path)

    console.print(theme.ok(
        f"Reverted {name} to v{version_num} (archived current as v{archive_ver})"
    ))
    console.print(theme.info(f"Use 'mosaicx template diff {name} --version {archive_ver}' to compare"))


@template.command("diff")
@click.argument("name")
@click.option("--version", "version_num", type=int, required=True, help="Version number to compare against current.")
def template_diff(name: str, version_num: int) -> None:
    """Show differences between current template and a previous version.

    NAME is the template name (e.g. MedicalReport).

    \b
    Examples:
      mosaicx template diff MedicalReport --version 1
    """
    try:
        import yaml
    except ImportError:
        raise click.ClickException("PyYAML required for template diff: pip install pyyaml")

    from .report import _find_user_template_yaml

    path = _find_user_template_yaml(name)
    if path is None:
        raise click.ClickException(f"User template {name!r} not found.")

    versions = _list_template_versions(name)
    version_paths = {v: p for v, p in versions}

    if version_num not in version_paths:
        available = ", ".join(f"v{v}" for v, _ in versions) or "none"
        raise click.ClickException(
            f"Version {version_num} not found for {name!r}. Available: {available}"
        )

    # Load both versions
    current_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    old_data = yaml.safe_load(
        version_paths[version_num].read_text(encoding="utf-8")
    )

    current_sections = {
        s["name"]: s for s in (current_data.get("sections") or [])
    }
    old_sections = {
        s["name"]: s for s in (old_data.get("sections") or [])
    }

    added = [n for n in current_sections if n not in old_sections]
    removed = [n for n in old_sections if n not in current_sections]
    modified = []
    for n in old_sections:
        if n in current_sections and old_sections[n] != current_sections[n]:
            modified.append(n)

    theme.section(f"{name}: v{version_num} vs current", console)

    if not added and not removed and not modified:
        # Check top-level metadata changes
        meta_changed = False
        for key in ("name", "description", "mode"):
            if current_data.get(key) != old_data.get(key):
                meta_changed = True
                console.print(theme.info(
                    f"  {key}: {old_data.get(key)!r} -> {current_data.get(key)!r}"
                ))
        if not meta_changed:
            console.print(theme.info("No differences"))
        return

    t = theme.make_clean_table()
    t.add_column("", no_wrap=True)
    t.add_column("Section", style=f"bold {theme.CORAL}", no_wrap=True)
    t.add_column("Detail", style=theme.MUTED)

    for n in added:
        s = current_sections[n]
        t.add_row("[green]+[/green]", n, f"({s.get('type', '?')})")
    for n in removed:
        s = old_sections[n]
        t.add_row("[red]-[/red]", n, f"({s.get('type', '?')})")
    for n in modified:
        changes = []
        os, ns = old_sections[n], current_sections[n]
        if os.get("type") != ns.get("type"):
            changes.append(f"type: {os.get('type')} -> {ns.get('type')}")
        if os.get("required") != ns.get("required"):
            changes.append(f"required changed")
        if os.get("description") != ns.get("description"):
            changes.append("description changed")
        t.add_row("[yellow]~[/yellow]", n, "; ".join(changes) or "content changed")

    console.print(Padding(t, (0, 0, 0, 2)))
    console.print()
    parts = []
    if added:
        parts.append(f"{len(added)} added")
    if removed:
        parts.append(f"{len(removed)} removed")
    if modified:
        parts.append(f"{len(modified)} modified")
    console.print(theme.info(", ".join(parts)))


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--document", type=click.Path(exists=True, path_type=Path), default=None, help="Single document to summarize.")
@click.option("--dir", "directory", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None, help="Directory of reports.")
@click.option("--patient", type=str, default=None, help="Patient identifier.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save output to file (.json or .yaml/.yml).")
@click.option("--format", "formats", type=str, multiple=True, help="Output format(s): json, jsonl, csv, parquet.")
def summarize(
    document: Optional[Path],
    directory: Optional[Path],
    patient: Optional[str],
    output: Optional[Path],
    formats: tuple[str, ...],
) -> None:
    """Summarize clinical reports for a patient.

    \b
    Produces a narrative summary and a timeline of key clinical events.
    Accepts a single file (--document) or a directory of reports (--dir).

    \b
    Examples:
      mosaicx summarize --document report.pdf
      mosaicx summarize --document report.pdf --patient "Patient A"
      mosaicx summarize --document report.pdf -o summary.json
      mosaicx summarize --dir reports/
    """
    if document is None and directory is None:
        raise click.ClickException("Provide --document or --dir.")

    # Preflight: check API key before expensive document loading
    _check_api_key()

    from .documents.models import DocumentLoadError

    # Collect report texts
    report_texts: list[str] = []

    if document is not None:
        try:
            doc = _load_doc_with_config(document)
        except (FileNotFoundError, ValueError, DocumentLoadError) as exc:
            raise click.ClickException(str(exc))
        if doc.quality_warning:
            console.print(theme.warn("Low OCR quality detected \u2014 results may be unreliable"))
        report_texts.append(doc.text)
    elif directory is not None:
        supported = (".txt", ".md", ".markdown", ".pdf")
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() in supported:
                try:
                    doc = _load_doc_with_config(p)
                    if doc.quality_warning:
                        console.print(theme.warn(
                            f"Low OCR quality: {p.name}"
                        ))
                    report_texts.append(doc.text)
                except Exception:
                    console.print(theme.warn(f"Skipping {p.name}: unsupported or unreadable"))

    if not report_texts:
        raise click.ClickException(
            "No reports found. --dir scans for .txt, .md, and .pdf files."
        )

    console.print(theme.info(f"Loaded {len(report_texts)} report(s)"))

    # Configure DSPy and run summarizer
    _configure_dspy()

    from .pipelines.summarizer import ReportSummarizer

    summarizer = ReportSummarizer()
    with theme.spinner("Summarizing... TL;DR incoming", console):
        result = summarizer(reports=report_texts, patient_id=patient or "")

    console.print(theme.ok("TL;DR ready \u2014 I see this as an absolute win"))

    # Build output data dict
    output_data = {
        "narrative": result.narrative,
        "events": [e.model_dump() for e in result.events] if result.events else [],
    }

    # Save to file if --output specified
    if output is not None:
        suffix = output.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise click.ClickException("PyYAML required for YAML output: pip install pyyaml")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(yaml.dump(output_data, default_flow_style=False, sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            if not suffix:
                output = output.with_suffix(".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(output_data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        console.print(theme.ok(f"Saved to {output}"))

    theme.section("Narrative Summary", console)
    console.print(Padding(result.narrative, (1, 2, 1, 4)))

    if result.events:
        theme.section("Timeline Events", console)
        t = Table(box=box.SIMPLE_HEAD, border_style=theme.GREIGE, padding=(0, 1))
        t.add_column("Date", style=f"bold {theme.CORAL}", no_wrap=True)
        t.add_column("Exam", style=f"bold")
        t.add_column("Key Finding")
        t.add_column("Change from Prior", style=f"{theme.MUTED}")
        for ev in result.events:
            t.add_row(
                ev.date,
                ev.exam_type,
                ev.key_finding,
                ev.change_from_prior or "",
            )
        console.print(Padding(t, (0, 0, 0, 2)))

    # Display performance metrics
    metrics = getattr(summarizer, "_last_metrics", None)
    if metrics is not None:
        from .cli_display import render_metrics
        render_metrics(metrics, console)


# ---------------------------------------------------------------------------
# deidentify
# ---------------------------------------------------------------------------


def _deidentify_batch(
    directory: Path,
    mode: str,
    regex_only: bool,
    output_dir_path: Optional[Path],
    formats: tuple[str, ...],
    workers: int,
    resume: bool,
) -> None:
    """Batch de-identify a directory of documents."""
    if not regex_only:
        _check_api_key()

    from .batch import BatchProcessor

    cfg = get_config()

    if output_dir_path is None:
        output_dir_path = directory.parent / f"{directory.name}_deidentified"

    effective_formats = formats if formats else ("json",)

    processor = BatchProcessor(
        workers=workers,
        checkpoint_every=cfg.checkpoint_every,
    )

    theme.section("Batch De-identification", console, "01")
    t = theme.make_clean_table(show_header=False)
    t.add_column("Key", style=f"bold {theme.CORAL}", no_wrap=True)
    t.add_column("Value")
    t.add_row("Input directory", str(directory))
    t.add_row("Output directory", str(output_dir_path))
    t.add_row("Mode", mode)
    t.add_row("Regex-only", theme.badge("Yes", "stable") if regex_only else "No")
    t.add_row("Export formats", ", ".join(effective_formats))
    t.add_row("Workers", str(workers))
    t.add_row("Resume", theme.badge("Yes", "stable") if resume else "No")
    console.print(Padding(t, (0, 0, 0, 2)))

    if regex_only:
        from .pipelines.deidentifier import regex_scrub_phi

        def process_fn(text: str) -> dict:
            return {"redacted_text": regex_scrub_phi(text), "mode": "regex"}
    else:
        _configure_dspy()
        from .pipelines.deidentifier import Deidentifier

        deid = Deidentifier()

        def process_fn(text: str) -> dict:
            result = deid(document_text=text, mode=mode)
            return {"redacted_text": result.redacted_text, "mode": mode}

    resume_id = "resume" if resume else None
    checkpoint_dir = output_dir_path / ".checkpoints" if resume else None

    supported = {".txt", ".md", ".pdf", ".docx"}
    total_docs = sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in supported)

    with theme.progress(total_docs, "documents", console) as advance:
        result = processor.process_directory(
            input_dir=directory,
            output_dir=output_dir_path,
            process_fn=process_fn,
            resume_id=resume_id,
            checkpoint_dir=checkpoint_dir,
            load_fn=lambda p: _load_doc_with_config(p).text,
            on_progress=lambda name, success: advance(),
        )

    console.print(theme.ok(f"Batch complete -- {result['succeeded']}/{result['total']} succeeded"))
    if result["skipped"]:
        console.print(theme.info(f"{result['skipped']} skipped (already processed)"))
    if result["failed"]:
        console.print(theme.warn(f"{result['failed']} failed"))
        for err in result.get("errors", []):
            console.print(theme.info(f"{err['file']}: {err['error']}"))

    # -- Export consolidated formats -----------------------------------------
    json_files = sorted(
        p for p in output_dir_path.glob("*.json")
        if not p.name.startswith(".")
    )

    if "jsonl" in effective_formats and json_files:
        jsonl_path = output_dir_path / "results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for jf in json_files:
                data = json.loads(jf.read_text(encoding="utf-8"))
                data["_source"] = jf.stem
                f.write(json.dumps(data, default=str, ensure_ascii=False) + "\n")
        console.print(theme.ok(f"Exported {jsonl_path}"))

    if "csv" in effective_formats and json_files:
        try:
            import pandas as pd

            records = []
            for jf in json_files:
                data = json.loads(jf.read_text(encoding="utf-8"))
                data["_source"] = jf.stem
                records.append(data)
            df = pd.json_normalize(records, sep="_")
            csv_path = output_dir_path / "results.csv"
            df.to_csv(csv_path, index=False)
            console.print(theme.ok(f"Exported {csv_path}"))
        except ImportError:
            console.print(theme.warn(
                "pandas required for CSV export: pip install pandas"
            ))

    if "parquet" in effective_formats and json_files:
        try:
            import pandas as pd

            records = []
            for jf in json_files:
                data = json.loads(jf.read_text(encoding="utf-8"))
                data["_source"] = jf.stem
                records.append(data)
            df = pd.json_normalize(records, sep="_")
            parquet_path = output_dir_path / "results.parquet"
            df.to_parquet(parquet_path, index=False)
            console.print(theme.ok(f"Exported {parquet_path}"))
        except ImportError:
            console.print(theme.warn(
                "pandas + pyarrow required for parquet: pip install pandas pyarrow"
            ))


@cli.command()
@click.option("--document", type=click.Path(exists=True, path_type=Path), help="Single document to de-identify.")
@click.option("--dir", "directory", type=click.Path(exists=True, file_okay=False, path_type=Path), help="Directory of documents.")
@click.option(
    "--mode",
    type=click.Choice(["remove", "pseudonymize", "dateshift"], case_sensitive=False),
    default="remove",
    show_default=True,
    help="De-identification strategy.",
)
@click.option("--regex-only", is_flag=True, default=False, help="Use regex-only PHI scrubbing (no LLM needed).")
@click.option("--workers", type=int, default=1, show_default=True, help="Number of parallel workers.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Save output to file (.json or .yaml/.yml) for single-file de-identification.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, help="Output directory for batch results (used with --dir).")
@click.option("--format", "formats", type=click.Choice(["json", "jsonl", "csv", "parquet"], case_sensitive=False), multiple=True, default=("json",), show_default=True, help="Output format(s) for batch results.")
@click.option("--resume", is_flag=True, default=False, help="Resume batch processing from last checkpoint.")
def deidentify(
    document: Optional[Path],
    directory: Optional[Path],
    mode: str,
    regex_only: bool,
    workers: int,
    output: Optional[Path],
    output_dir: Optional[Path],
    formats: tuple[str, ...],
    resume: bool,
) -> None:
    """De-identify clinical documents by removing or replacing PHI.

    \b
    Accepts a single file (--document) or a directory (--dir).
    Use --regex-only for fast rule-based scrubbing without an LLM.

    \b
    Examples:
      mosaicx deidentify --document note.pdf
      mosaicx deidentify --document note.pdf --mode pseudonymize
      mosaicx deidentify --document note.pdf --regex-only -o clean.json
      mosaicx deidentify --dir notes/ --workers 4 --output-dir cleaned/
    """
    if document is None and directory is None:
        raise click.ClickException("Provide --document or --dir.")
    if document is not None and directory is not None:
        raise click.UsageError("--document and --dir are mutually exclusive.")

    # Route batch to helper
    if directory is not None:
        _deidentify_batch(
            directory=directory,
            mode=mode,
            regex_only=regex_only,
            output_dir_path=output_dir,
            formats=formats,
            workers=workers,
            resume=resume,
        )
        return

    # Single file processing
    if not regex_only:
        _check_api_key()

    from .documents.models import DocumentLoadError

    try:
        doc = _load_doc_with_config(document)
    except (FileNotFoundError, ValueError, DocumentLoadError) as exc:
        raise click.ClickException(str(exc))

    if doc.quality_warning:
        console.print(theme.warn("Low OCR quality detected -- results may be unreliable"))

    console.print(theme.info(f"De-identifying 1 document -- mode: {mode}{'  -- regex-only' if regex_only else ''}"))

    if regex_only:
        from .pipelines.deidentifier import regex_scrub_phi

        redacted = regex_scrub_phi(doc.text)
    else:
        _configure_dspy()
        from .pipelines.deidentifier import Deidentifier

        deid = Deidentifier()
        with theme.spinner(f"Scrubbing {document.name}... nothing to see here", console):
            result = deid(document_text=doc.text, mode=mode)
        redacted = result.redacted_text
        console.print(theme.ok("Scrubbed -- PHI has left the chat"))

    # Save if --output
    if output is not None:
        save_data = {"redacted_text": redacted, "mode": "regex" if regex_only else mode}
        suffix = output.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise click.ClickException(
                    "PyYAML required for YAML output: pip install pyyaml"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                yaml.dump(
                    save_data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        else:
            if not suffix:
                output = output.with_suffix(".json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(save_data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
        console.print(theme.ok(f"Saved to {output}"))

    # Display
    theme.section(document.name, console, uppercase=False)
    console.print(
        Panel(
            redacted,
            box=box.ROUNDED,
            border_style=theme.GREIGE,
            padding=(1, 2),
        )
    )

    # Metrics (LLM path only)
    if not regex_only:
        doc_metrics = getattr(deid, "_last_metrics", None)
        if doc_metrics is not None:
            from .cli_display import render_metrics

            render_metrics(doc_metrics, console)


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--pipeline", type=str, default=None, help="Pipeline to optimize.")
@click.option("--trainset", type=click.Path(exists=True, path_type=Path), help="Training dataset path.")
@click.option("--valset", type=click.Path(exists=True, path_type=Path), help="Validation dataset path.")
@click.option(
    "--budget",
    type=click.Choice(["light", "medium", "heavy"], case_sensitive=False),
    default="medium",
    show_default=True,
    help="Optimization budget preset.",
)
@click.option("--save", type=click.Path(path_type=Path), help="Save optimized program to this path.")
@click.option("--list-pipelines", is_flag=True, default=False, help="List available pipelines and exit.")
def optimize(
    pipeline: Optional[str],
    trainset: Optional[Path],
    valset: Optional[Path],
    budget: str,
    save: Optional[Path],
    list_pipelines: bool,
) -> None:
    """Optimize a DSPy pipeline with labeled examples.

    \b
    Tunes pipeline prompts using a training set. Use --list-pipelines
    to see available pipelines.

    \b
    Examples:
      mosaicx optimize --pipeline radiology --trainset train.jsonl
      mosaicx optimize --pipeline radiology --trainset train.jsonl --budget heavy
      mosaicx optimize --pipeline radiology --trainset train.jsonl --save optimized.json
      mosaicx optimize --list-pipelines
    """
    from .evaluation.optimize import (
        OPTIMIZATION_STRATEGY,
        get_optimizer_config,
        get_pipeline_class,
        list_pipelines as _list_pipelines,
        run_optimization,
    )

    # --list-pipelines: show available pipelines and exit
    if list_pipelines:
        theme.section("Available Pipelines", console, "01")
        for name in _list_pipelines():
            console.print(f"  [{theme.CORAL}]>[/{theme.CORAL}] [{theme.GREIGE}]{name}[/{theme.GREIGE}]")
        return

    # Validate required args early
    if not pipeline:
        raise click.ClickException(
            "--pipeline is required. Use --list-pipelines to see available options."
        )

    if trainset is None:
        raise click.ClickException("--trainset is required.")

    try:
        opt_config = get_optimizer_config(budget)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # 01 · Configuration display
    theme.section("Optimization", console, "01")
    t = theme.make_kv_table()
    t.add_row("Pipeline", pipeline)
    t.add_row("Budget", theme.badge(budget.upper()))
    t.add_row("Strategy", opt_config.get("strategy", "N/A"))
    t.add_row("Max iterations", str(opt_config.get("max_iterations", "N/A")))
    t.add_row("Num candidates", str(opt_config.get("num_candidates", "N/A")))
    t.add_row("Training set", str(trainset))
    t.add_row("Validation set", str(valset) if valset else "[dim]not specified[/dim]")
    t.add_row("Save path", str(save) if save else "[dim]not specified[/dim]")
    console.print(t)

    # Validate pipeline name
    try:
        pipeline_cls = get_pipeline_class(pipeline)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # Load dataset
    from .evaluation.dataset import load_jsonl
    from .evaluation.metrics import get_metric

    try:
        train_examples = load_jsonl(trainset, pipeline)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc))

    val_examples = None
    if valset:
        try:
            val_examples = load_jsonl(valset, pipeline)
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc))

    try:
        metric = get_metric(pipeline)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    theme.section("Dataset", console, "02")
    console.print(theme.info(f"Loaded {len(train_examples)} training examples"))
    if val_examples:
        console.print(theme.info(f"Loaded {len(val_examples)} validation examples"))

    # Configure DSPy
    _configure_dspy()

    # Determine save path
    if save is None:
        cfg = get_config()
        save = cfg.optimized_dir / f"{pipeline}_optimized.json"

    # Run optimization
    theme.section("Running Optimization", console, "03")
    module = pipeline_cls()
    with theme.spinner("Optimizing... patience you must have", console):
        optimized, results = run_optimization(
            module=module,
            trainset=train_examples,
            valset=val_examples,
            metric=metric,
            budget=budget,
            save_path=save,
        )

    # 04 · Results
    theme.section("Results", console, "04")
    results_table = theme.make_kv_table()
    results_table.add_row("Strategy", results["strategy"])
    results_table.add_row("Train examples", str(results["num_train"]))
    results_table.add_row("Val examples", str(results["num_val"]))
    results_table.add_row("Train score", f"{results['train_score']:.3f}")
    results_table.add_row("Val score", f"{results['val_score']:.3f}")
    results_table.add_row("Saved to", str(save))
    console.print(results_table)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@cli.command(name="eval")
@click.option("--pipeline", type=str, required=True, help="Pipeline to evaluate.")
@click.option("--testset", type=click.Path(exists=True, path_type=Path), required=True, help="Test dataset path.")
@click.option("--optimized", type=click.Path(exists=True, path_type=Path), default=None, help="Path to optimized program.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Save detailed results JSON to this path.")
def eval_cmd(
    pipeline: str,
    testset: Path,
    optimized: Optional[Path],
    output: Optional[Path],
) -> None:
    """Evaluate a pipeline against a labeled test set.

    \b
    Runs the pipeline on each example in the test set and reports
    accuracy metrics. Optionally use an optimized program.

    \b
    Examples:
      mosaicx eval --pipeline radiology --testset test.jsonl
      mosaicx eval --pipeline radiology --testset test.jsonl --optimized opt.json
      mosaicx eval --pipeline radiology --testset test.jsonl --output results.json
    """
    from .evaluation.dataset import load_jsonl
    from .evaluation.metrics import get_metric
    from .evaluation.optimize import get_pipeline_class, load_optimized

    # Validate pipeline
    try:
        pipeline_cls = get_pipeline_class(pipeline)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # Load test set
    try:
        test_examples = load_jsonl(testset, pipeline)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc))

    try:
        metric = get_metric(pipeline)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    theme.section("Evaluation", console, "01")
    t = theme.make_kv_table()
    t.add_row("Pipeline", pipeline)
    t.add_row("Test set", str(testset))
    t.add_row("Examples", str(len(test_examples)))
    t.add_row("Optimized", str(optimized) if optimized else "[dim]baseline[/dim]")
    console.print(t)

    # Configure DSPy and instantiate module
    _configure_dspy()

    if optimized:
        module = load_optimized(pipeline_cls, optimized)
        console.print(theme.info(f"Loaded optimized program from {optimized}"))
    else:
        module = pipeline_cls()
        console.print(theme.info("Evaluating baseline (unoptimized) program"))

    # Run each example through pipeline
    import statistics

    scores: list[float] = []
    details: list[dict] = []

    with theme.spinner("Evaluating... patience you must have", console):
        for i, example in enumerate(test_examples):
            try:
                prediction = module(**dict(example.inputs()))
                score = metric(example, prediction)
            except Exception as exc:
                score = 0.0
                prediction = None
                console.print(theme.warn(f"Example {i+1} failed: {exc}"))

            scores.append(score)
            details.append({
                "index": i,
                "score": score,
                "inputs": {k: str(v)[:200] for k, v in example.inputs().items()},
            })

    # 02 · Statistics
    theme.section("Statistics", console, "02")
    stats_table = theme.make_kv_table()
    stats_table.add_row("Count", str(len(scores)))
    stats_table.add_row("Mean", f"{statistics.mean(scores):.3f}")
    stats_table.add_row("Median", f"{statistics.median(scores):.3f}")
    if len(scores) >= 2:
        stats_table.add_row("Std Dev", f"{statistics.stdev(scores):.3f}")
    stats_table.add_row("Min", f"{min(scores):.3f}")
    stats_table.add_row("Max", f"{max(scores):.3f}")
    console.print(stats_table)

    # 03 · Score distribution histogram (unicode bar chart)
    theme.section("Score Distribution", console, "03")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_counts = [0] * (len(bins) - 1)
    for s in scores:
        for j in range(len(bins) - 1):
            if bins[j] <= s < bins[j + 1] or (j == len(bins) - 2 and s == bins[j + 1]):
                bin_counts[j] += 1
                break

    max_count = max(bin_counts) if bin_counts else 1
    bar_width = 30
    hist_table = theme.make_table()
    hist_table.add_column("Range", style="bold cyan", no_wrap=True)
    hist_table.add_column("Count", style="magenta", justify="right")
    hist_table.add_column("Distribution")

    for j in range(len(bins) - 1):
        label = f"{bins[j]:.1f}-{bins[j+1]:.1f}"
        count = bin_counts[j]
        bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        hist_table.add_row(label, str(count), f"[cyan]{bar}[/cyan]")
    console.print(hist_table)

    # Save detailed results
    if output:
        output_data = {
            "pipeline": pipeline,
            "testset": str(testset),
            "optimized": str(optimized) if optimized else None,
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) >= 2 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "details": details,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(output_data, indent=2))
        console.print(theme.info(f"Results saved to {output}"))


# ---------------------------------------------------------------------------
# config (group)
# ---------------------------------------------------------------------------


@cli.group()
def config() -> None:
    """View and update MOSAICX configuration."""


@config.command("show")
def config_show() -> None:
    """Show current configuration.

    \b
    Examples:
      mosaicx config show
    """
    cfg = get_config()
    dump = cfg.model_dump()

    # 01 · Language Models
    theme.section("Language Models", console, "01")
    t = theme.make_kv_table()
    t.add_row("lm", dump["lm"])
    t.add_row("lm_cheap", dump["lm_cheap"])
    t.add_row("api_base", dump["api_base"])
    t.add_row("lm_temperature", str(dump["lm_temperature"]))
    api_key = dump["api_key"]
    if api_key:
        masked = api_key[:4] + "\u00b7\u00b7\u00b7" + api_key[-4:] if len(api_key) > 8 else "***"
    else:
        masked = "[dim]not set[/dim]"
    t.add_row("api_key", masked)
    console.print(t)

    # 02 · Processing
    theme.section("Processing", console, "02")
    t = theme.make_kv_table()
    t.add_row("default_template", dump["default_template"])
    t.add_row("completeness_threshold", str(dump["completeness_threshold"]))
    t.add_row("batch_workers", str(dump["batch_workers"]))
    t.add_row("checkpoint_every", str(dump["checkpoint_every"]))
    console.print(t)

    # 03 · Document OCR
    theme.section("Document OCR", console, "03")
    t = theme.make_kv_table()
    t.add_row("ocr_engine", theme.badge(dump["ocr_engine"].upper()))
    t.add_row("chandra_backend", dump["chandra_backend"])
    if dump.get("chandra_server_url"):
        t.add_row("chandra_server_url", dump["chandra_server_url"])
    t.add_row("quality_threshold", str(dump["quality_threshold"]))
    t.add_row("ocr_page_timeout", f"{dump['ocr_page_timeout']}s")
    t.add_row("force_ocr", str(dump["force_ocr"]))
    t.add_row("ocr_langs", ", ".join(dump["ocr_langs"]))
    console.print(t)

    # 04 · Export & Privacy
    theme.section("Export & Privacy", console, "04")
    t = theme.make_kv_table()
    t.add_row("export_formats", ", ".join(dump["default_export_formats"]))
    t.add_row("deidentify_mode", dump["deidentify_mode"])
    console.print(t)

    # 05 · Paths
    theme.section("Paths", console, "05")
    t = theme.make_kv_table()
    t.add_row("home_dir", str(dump["home_dir"]))
    t.add_row("templates_dir", str(cfg.templates_dir))
    t.add_row("schema_dir", str(cfg.schema_dir))
    t.add_row("optimized_dir", str(cfg.optimized_dir))
    t.add_row("checkpoint_dir", str(cfg.checkpoint_dir))
    t.add_row("log_dir", str(cfg.log_dir))
    console.print(t)
    console.print()


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value.

    Not yet implemented -- use environment variables instead.

    \b
    KEY is the config key (e.g. lm, api_base, lm_temperature).
    VALUE is the value to set.

    \b
    Examples:
      mosaicx config set lm ollama_chat/llama3.1
      mosaicx config set api_base http://localhost:11434
    """
    env_var = f"MOSAICX_{key.upper()}"
    console.print(theme.warn(
        "Config persistence is not implemented yet. "
        "Set values via environment variables or a .env file:"
    ))
    console.print(theme.info(f"  export {env_var}={value}"))


# ---------------------------------------------------------------------------
# mcp (group)
# ---------------------------------------------------------------------------


@cli.group()
def mcp() -> None:
    """Model Context Protocol (MCP) server for AI agents."""


@mcp.command("serve")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"], case_sensitive=False),
    default="stdio",
    show_default=True,
    help="MCP transport protocol.",
)
@click.option(
    "--port",
    type=int,
    default=8080,
    show_default=True,
    help="Port for SSE transport (ignored for stdio).",
)
def mcp_serve(transport: str, port: int) -> None:
    """Start the MOSAICX MCP server.

    \b
    Exposes MOSAICX pipelines as MCP tools that AI agents (Claude, etc.)
    can call directly. Use stdio for Claude Code/Desktop integration,
    or sse for HTTP-based clients.

    \b
    Examples:
      mosaicx mcp serve
      mosaicx mcp serve --transport sse --port 9090
    """
    try:
        from .mcp_server import mcp as mcp_server
    except SystemExit:
        raise click.ClickException(
            "The 'mcp' package is required. Install with: pip install 'mosaicx[mcp]'"
        )

    cfg = get_config()
    console.print(theme.info(f"Starting MOSAICX MCP server (transport: {transport})"))
    console.print(theme.info(f"Model: {cfg.lm}"))
    console.print(theme.info("Tools: extract_document, deidentify_text, generate_template, list_templates, list_modes"))

    if transport == "sse":
        mcp_server.run(transport="sse", port=port)
    else:
        mcp_server.run(transport="stdio")

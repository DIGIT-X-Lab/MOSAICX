# mosaicx/cli.py
"""
MOSAICX CLI v2 -- Click commands with DigiTx-inspired terminal UI.

Provides the ``mosaicx`` console entry-point declared in pyproject.toml as
``mosaicx.cli:cli``.  Commands call into the actual pipeline modules:

- extract:    DocumentExtractor (DSPy)
- batch:      BatchProcessor
- template:   radreport registry + template_compiler
- schema:     SchemaGenerator (DSPy)
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
from typing import Optional

import click
from rich import box
from rich.console import Console
from rich.markup import escape as _esc
from rich.padding import Padding
from rich.panel import Panel

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
    dspy.configure(lm=dspy.LM(cfg.lm, api_key=cfg.api_key, api_base=cfg.api_base))


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
    theme.print_banner(_VERSION_NUMBER, console, lm=cfg.lm, lm_cheap=cfg.lm_cheap)
    if ctx.invoked_subcommand is None:
        console.print()
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--document", type=click.Path(exists=False, path_type=Path), help="Path to clinical document.")
@click.option("--template", type=str, default="auto", help="Template name or path.")
@click.option("--optimized", type=click.Path(exists=False, path_type=Path), default=None, help="Path to optimized program.")
def extract(
    document: Optional[Path],
    template: str,
    optimized: Optional[Path],
) -> None:
    """Extract structured data from a clinical document."""
    if document is None:
        raise click.ClickException("--document is required.")

    # Load the document
    from .documents.models import DocumentLoadError

    try:
        doc = _load_doc_with_config(document)
    except (FileNotFoundError, ValueError, DocumentLoadError) as exc:
        raise click.ClickException(str(exc))

    if doc.quality_warning:
        console.print(theme.warn("Low OCR quality detected \u2014 results may be unreliable"))

    if doc.is_empty:
        raise click.ClickException(f"Document is empty: {document}")

    # Build a descriptive load line
    parts = [f"{doc.format} document", f"{doc.char_count:,} chars"]
    if doc.ocr_engine_used:
        parts.append(f"OCR: {doc.ocr_engine_used}")
    if doc.ocr_confidence:
        parts.append(f"confidence: {doc.ocr_confidence:.0%}")
    console.print(theme.info(" \u00b7 ".join(parts)))

    # Resolve template to an output schema if not "auto"
    output_schema = None
    if template != "auto":
        template_path = Path(template)
        if template_path.exists() and template_path.suffix in (".yaml", ".yml"):
            from .schemas.template_compiler import compile_template_file

            try:
                output_schema = compile_template_file(template_path)
                console.print(theme.info(f"Using template: {template_path.name}"))
            except Exception as exc:
                raise click.ClickException(f"Failed to compile template: {exc}")
        else:
            # Try registry lookup
            from .schemas.radreport.registry import get_template

            tpl_info = get_template(template)
            if tpl_info is not None:
                console.print(theme.info(
                    f"Using built-in template: {tpl_info.name} ({tpl_info.description})"
                ))
            else:
                raise click.ClickException(
                    f"Template not found: {template!r}. "
                    "Provide a .yaml file path or a registered template name."
                )

    # Configure DSPy and run extraction
    _configure_dspy()

    from .pipelines.extraction import DocumentExtractor

    extractor = DocumentExtractor(output_schema=output_schema)

    if optimized is not None:
        from .evaluation.optimize import load_optimized

        extractor = load_optimized(type(extractor), optimized)

    with theme.spinner("Extracting... patience you must have", console):
        result = extractor(document_text=doc.text)

    console.print(theme.ok("Extracted \u2014 this is the way"))

    # Format output as JSON
    output: dict = {}
    if hasattr(result, "extracted"):
        output["extracted"] = result.extracted.model_dump() if hasattr(result.extracted, "model_dump") else str(result.extracted)
    else:
        if hasattr(result, "demographics"):
            output["demographics"] = result.demographics.model_dump()
        if hasattr(result, "findings"):
            output["findings"] = [f.model_dump() for f in result.findings]
        if hasattr(result, "diagnoses"):
            output["diagnoses"] = [d.model_dump() for d in result.diagnoses]

    theme.section("Extracted Data", console)
    from rich.json import JSON

    console.print(Padding(JSON(json.dumps(output, default=str)), (0, 0, 0, 2)))


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--input-dir", type=click.Path(exists=False, path_type=Path), help="Directory of input documents.")
@click.option("--output-dir", type=click.Path(path_type=Path), help="Directory for output files.")
@click.option("--template", type=str, default="auto", help="Template name or path.")
@click.option("--format", "formats", type=str, multiple=True, help="Output format(s).")
@click.option("--workers", type=int, default=4, show_default=True, help="Number of parallel workers.")
@click.option("--completeness-threshold", type=float, default=0.7, show_default=True, help="Minimum completeness score.")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint.")
def batch(
    input_dir: Optional[Path],
    output_dir: Optional[Path],
    template: str,
    formats: tuple[str, ...],
    workers: int,
    completeness_threshold: float,
    resume: bool,
) -> None:
    """Batch-process a directory of documents."""
    if input_dir is None:
        raise click.ClickException("--input-dir is required.")
    if output_dir is None:
        raise click.ClickException("--output-dir is required.")
    if not input_dir.is_dir():
        raise click.ClickException(f"Input directory does not exist: {input_dir}")

    from .batch import BatchProcessor

    cfg = get_config()
    effective_formats = formats if formats else tuple(cfg.default_export_formats)
    effective_workers = workers
    effective_threshold = completeness_threshold

    processor = BatchProcessor(
        workers=effective_workers,
        checkpoint_every=cfg.checkpoint_every,
    )

    theme.section("Batch Processing", console, "01")
    t = theme.make_kv_table()
    t.add_row("Input directory", str(input_dir))
    t.add_row("Output directory", str(output_dir))
    t.add_row("Template", template)
    t.add_row("Export formats", ", ".join(effective_formats))
    t.add_row("Workers", str(effective_workers))
    t.add_row("Completeness", str(effective_threshold))
    t.add_row("Resume", theme.badge("Yes", "stable") if resume else "No")
    console.print(t)

    resume_id = "resume" if resume else None
    with theme.spinner("Deploying the minions...", console):
        result = processor.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            process_fn=lambda doc: doc,  # placeholder until full wiring
            template=template,
            resume_id=resume_id,
        )
    console.print(theme.ok("Batch complete \u2014 it ain't much but it's honest work"))


# ---------------------------------------------------------------------------
# template (group)
# ---------------------------------------------------------------------------


@cli.group()
def template() -> None:
    """Manage extraction templates."""


@template.command("create")
def template_create() -> None:
    """Create a new extraction template."""
    console.print(theme.info("template create \u2014 not yet implemented"))


@template.command("list")
def template_list() -> None:
    """List available templates."""
    from .schemas.radreport.registry import list_templates

    templates = list_templates()

    theme.section("Templates", console)

    t = theme.make_table()
    t.add_column("Name", style="bold cyan", no_wrap=True)
    t.add_column("Exam Type", style="magenta")
    t.add_column("RadReport ID", style="dim")
    t.add_column("Description")

    for tpl in templates:
        t.add_row(
            tpl.name,
            tpl.exam_type,
            tpl.radreport_id or "\u2014",
            tpl.description,
        )

    console.print(t)
    console.print(theme.info(f"{len(templates)} template(s) registered"))


@template.command("validate")
@click.option("--file", "file_path", type=click.Path(exists=False, path_type=Path), required=True, help="Template YAML file to validate.")
def template_validate(file_path: Path) -> None:
    """Validate a template file."""
    from .schemas.template_compiler import compile_template_file

    if not file_path.exists():
        raise click.ClickException(f"File not found: {file_path}")

    try:
        model = compile_template_file(file_path)
        console.print(theme.ok("Template is valid \u2014 you shall pass"))
        console.print(theme.info(f"Model: {model.__name__}"))
        console.print(theme.info(f"Fields: {', '.join(model.model_fields.keys())}"))
    except Exception as exc:
        raise click.ClickException(f"Template validation failed: {exc}")


# ---------------------------------------------------------------------------
# schema (group)
# ---------------------------------------------------------------------------


@cli.group()
def schema() -> None:
    """Manage Pydantic schemas."""


@schema.command("generate")
@click.option("--description", type=str, required=True, help="Natural-language description of the schema.")
@click.option("--name", type=str, default=None, help="Schema class name (default: LLM-chosen).")
@click.option("--example-text", type=str, default="", help="Optional example document text for grounding.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Save schema to this path (default: ~/.mosaicx/schemas/).")
def schema_generate(description: str, name: Optional[str], example_text: str, output: Optional[Path]) -> None:
    """Generate a Pydantic schema from a description."""
    _configure_dspy()

    from .pipelines.schema_gen import SchemaGenerator, save_schema

    generator = SchemaGenerator()
    with theme.spinner("Generating schema... hold my beer", console):
        result = generator(description=description, example_text=example_text)

    # Override class_name if user specified --name
    if name:
        result.schema_spec.class_name = name

    # Save schema
    cfg = get_config()
    saved_path = save_schema(
        result.schema_spec,
        schema_dir=None if output else cfg.schema_dir,
        output_path=output,
    )

    console.print(theme.ok("Schema generated \u2014 it's alive!"))
    console.print(theme.info(f"Model: {result.compiled_model.__name__}"))
    console.print(theme.info(
        f"Fields: {', '.join(result.compiled_model.model_fields.keys())}"
    ))
    console.print(theme.info(f"Saved: {saved_path}"))

    theme.section("Schema Spec", console)
    from rich.json import JSON

    console.print(Padding(JSON(result.schema_spec.model_dump_json()), (0, 0, 0, 2)))


@schema.command("list")
def schema_list() -> None:
    """List saved schemas."""
    from .pipelines.schema_gen import list_schemas

    cfg = get_config()
    specs = list_schemas(cfg.schema_dir)

    theme.section("Saved Schemas", console)

    if not specs:
        console.print(theme.info("0 schema(s) saved"))
        return

    t = theme.make_clean_table(width=len(theme.TAGLINE))
    t.add_column("Name", style=f"bold {theme.CORAL}", no_wrap=True)
    t.add_column("Fields", style=f"{theme.GREIGE}", justify="right")
    t.add_column("Description", style=theme.MUTED, ratio=1)

    for spec in specs:
        t.add_row(
            spec.class_name,
            str(len(spec.fields)),
            spec.description if spec.description else "\u2014",
        )

    console.print(Padding(t, (0, 0, 0, 2)))
    console.print()
    console.print(theme.info(f"{len(specs)} schema(s) saved in {cfg.schema_dir}"))


@schema.command("refine")
@click.option("--schema", "schema_name", type=str, required=True, help="Name of the schema to refine.")
@click.option("--instruction", type=str, default=None, help="Natural-language refinement instruction (uses LLM).")
@click.option("--add", "add_field_str", type=str, default=None, help="Add a field: 'field_name: type'.")
@click.option("--remove", "remove_field_name", type=str, default=None, help="Remove a field by name.")
@click.option("--rename", "rename_str", type=str, default=None, help="Rename a field: 'old_name=new_name'.")
def schema_refine(
    schema_name: str,
    instruction: Optional[str],
    add_field_str: Optional[str],
    remove_field_name: Optional[str],
    rename_str: Optional[str],
) -> None:
    """Refine an existing schema."""
    from .pipelines.schema_gen import (
        load_schema, save_schema, compile_schema,
        add_field, remove_field, rename_field,
    )

    cfg = get_config()

    try:
        spec = load_schema(schema_name, cfg.schema_dir)
    except FileNotFoundError:
        raise click.ClickException(f"Schema not found: {schema_name}")

    old_field_names = {f.name for f in spec.fields}

    if instruction:
        # LLM-driven refinement
        _configure_dspy()
        from .pipelines.schema_gen import SchemaRefiner

        refiner = SchemaRefiner()
        with theme.spinner("Refining schema... one does not simply edit a schema", console):
            result = refiner(
                current_schema=spec.model_dump_json(indent=2),
                instruction=instruction,
            )
        spec = result.schema_spec

    elif add_field_str:
        # --add "field_name: type"
        parts = add_field_str.split(":", 1)
        if len(parts) != 2:
            raise click.ClickException("--add format: 'field_name: type'")
        fname, ftype = parts[0].strip(), parts[1].strip()
        spec = add_field(spec, fname, ftype)

    elif remove_field_name:
        try:
            spec = remove_field(spec, remove_field_name)
        except ValueError as exc:
            raise click.ClickException(str(exc))

    elif rename_str:
        parts = rename_str.split("=", 1)
        if len(parts) != 2:
            raise click.ClickException("--rename format: 'old_name=new_name'")
        old, new = parts[0].strip(), parts[1].strip()
        try:
            spec = rename_field(spec, old, new)
        except ValueError as exc:
            raise click.ClickException(str(exc))

    else:
        raise click.ClickException(
            "Provide --instruction, --add, --remove, or --rename"
        )

    # Verify the schema compiles
    try:
        compiled = compile_schema(spec)
    except Exception as exc:
        raise click.ClickException(f"Refined schema failed to compile: {exc}")

    # Save back
    save_schema(spec, schema_dir=cfg.schema_dir)

    # Show changes
    new_field_names = {f.name for f in spec.fields}
    added = new_field_names - old_field_names
    removed = old_field_names - new_field_names

    console.print(theme.ok("Schema refined \u2014 evolution, not revolution"))

    if added:
        for name in sorted(added):
            f = next(f for f in spec.fields if f.name == name)
            console.print(theme.info(f"+ {name} ({f.type})"))
    if removed:
        for name in sorted(removed):
            console.print(theme.info(f"- {name} (removed)"))

    # Show renamed (detected by same position, different name)
    if rename_str and "=" in rename_str:
        old_n, new_n = rename_str.split("=", 1)
        console.print(theme.info(f"~ {old_n.strip()} \u2192 {new_n.strip()}"))

    console.print(theme.info(f"Model: {compiled.__name__}"))
    console.print(theme.info(
        f"Fields: {', '.join(compiled.model_fields.keys())}"
    ))


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--document", type=click.Path(exists=False, path_type=Path), default=None, help="Single document to summarize.")
@click.option("--dir", "directory", type=click.Path(exists=False, path_type=Path), help="Directory of reports.")
@click.option("--patient", type=str, default=None, help="Patient identifier.")
@click.option("--format", "formats", type=str, multiple=True, help="Output format(s).")
def summarize(
    document: Optional[Path],
    directory: Optional[Path],
    patient: Optional[str],
    formats: tuple[str, ...],
) -> None:
    """Summarize clinical reports for a patient."""
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
        if not directory.is_dir():
            raise click.ClickException(f"Directory not found: {directory}")
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() in (".txt", ".md", ".markdown"):
                try:
                    doc = _load_doc_with_config(p)
                    if doc.quality_warning:
                        console.print(theme.warn(
                            f"Low OCR quality: {p.name}"
                        ))
                    report_texts.append(doc.text)
                except Exception:
                    console.print(theme.warn(f"Skipping {p.name}: unsupported or unreadable"))
    else:
        raise click.ClickException("Provide --document or --dir.")

    if not report_texts:
        raise click.ClickException("No reports found to summarize.")

    console.print(theme.info(f"Loaded {len(report_texts)} report(s)"))

    # Configure DSPy and run summarizer
    _configure_dspy()

    from .pipelines.summarizer import ReportSummarizer

    summarizer = ReportSummarizer()
    with theme.spinner("Summarizing... TL;DR incoming", console):
        result = summarizer(reports=report_texts, patient_id=patient or "")

    console.print(theme.ok("TL;DR ready \u2014 I see this as an absolute win"))

    theme.section("Narrative Summary", console)
    console.print(
        Panel(
            result.narrative,
            box=box.ROUNDED,
            border_style=theme.GREIGE,
            padding=(1, 2),
        )
    )

    if result.events:
        console.print(theme.info(f"{len(result.events)} timeline event(s) extracted"))


# ---------------------------------------------------------------------------
# deidentify
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--document", type=click.Path(exists=False, path_type=Path), help="Single document to de-identify.")
@click.option("--dir", "directory", type=click.Path(exists=False, path_type=Path), help="Directory of documents.")
@click.option(
    "--mode",
    type=click.Choice(["remove", "pseudonymize", "dateshift"], case_sensitive=False),
    default="remove",
    show_default=True,
    help="De-identification strategy.",
)
@click.option("--regex-only", is_flag=True, default=False, help="Use regex-only PHI scrubbing (no LLM needed).")
@click.option("--workers", type=int, default=4, show_default=True, help="Number of parallel workers.")
def deidentify(
    document: Optional[Path],
    directory: Optional[Path],
    mode: str,
    regex_only: bool,
    workers: int,
) -> None:
    """De-identify clinical documents."""
    from .pipelines.deidentifier import regex_scrub_phi

    if document is None and directory is None:
        raise click.ClickException("Provide --document or --dir.")

    # Collect file paths
    paths: list[Path] = []
    if document is not None:
        if not document.exists():
            raise click.ClickException(f"Document not found: {document}")
        paths.append(document)
    elif directory is not None:
        if not directory.is_dir():
            raise click.ClickException(f"Directory not found: {directory}")
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() in (".txt", ".md", ".markdown"):
                paths.append(p)

    if not paths:
        raise click.ClickException("No documents found to de-identify.")

    console.print(theme.info(
        f"De-identifying {len(paths)} document(s) \u00b7 mode: {mode}"
        f"{' \u00b7 regex-only' if regex_only else ''}"
    ))

    if regex_only:
        # Regex-only mode: no LLM needed
        for p in paths:
            text = p.read_text(encoding="utf-8")
            scrubbed = regex_scrub_phi(text)
            theme.section(p.name, console, uppercase=False)
            console.print(
                Panel(
                    scrubbed,
                    box=box.ROUNDED,
                    border_style=theme.GREIGE,
                    padding=(1, 2),
                )
            )
    else:
        # Full LLM + regex pipeline
        _configure_dspy()

        from .documents.models import DocumentLoadError
        from .pipelines.deidentifier import Deidentifier

        deid = Deidentifier()
        for p in paths:
            try:
                doc = _load_doc_with_config(p)
            except (FileNotFoundError, ValueError, DocumentLoadError) as exc:
                console.print(theme.warn(f"Skipping {p.name}: {exc}"))
                continue
            if doc.quality_warning:
                console.print(theme.warn("Low OCR quality detected \u2014 results may be unreliable"))
            with theme.spinner(f"Scrubbing {p.name}... nothing to see here", console):
                result = deid(document_text=doc.text, mode=mode)
            console.print(theme.ok("Scrubbed \u2014 PHI has left the chat"))
            theme.section(p.name, console, uppercase=False)
            console.print(
                Panel(
                    result.redacted_text,
                    box=box.ROUNDED,
                    border_style=theme.GREIGE,
                    padding=(1, 2),
                )
            )


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--pipeline", type=str, default=None, help="Pipeline to optimize.")
@click.option("--trainset", type=click.Path(exists=False, path_type=Path), help="Training dataset path.")
@click.option("--valset", type=click.Path(exists=False, path_type=Path), help="Validation dataset path.")
@click.option(
    "--budget",
    type=click.Choice(["light", "medium", "heavy"], case_sensitive=False),
    default="medium",
    show_default=True,
    help="Optimization budget preset.",
)
@click.option("--save", type=click.Path(path_type=Path), help="Save optimized program to this path.")
def optimize(
    pipeline: Optional[str],
    trainset: Optional[Path],
    valset: Optional[Path],
    budget: str,
    save: Optional[Path],
) -> None:
    """Optimize a DSPy pipeline."""
    from .evaluation.optimize import OPTIMIZATION_STRATEGY, get_optimizer_config

    try:
        opt_config = get_optimizer_config(budget)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    theme.section("Optimization", console, "01")
    t = theme.make_kv_table()
    t.add_row("Pipeline", pipeline or "[dim]not specified[/dim]")
    t.add_row("Budget", theme.badge(budget.upper()))
    t.add_row("Strategy", opt_config.get("strategy", "N/A"))
    t.add_row("Max iterations", str(opt_config.get("max_iterations", "N/A")))
    t.add_row("Num candidates", str(opt_config.get("num_candidates", "N/A")))
    t.add_row("Training set", str(trainset) if trainset else "[dim]not specified[/dim]")
    t.add_row("Validation set", str(valset) if valset else "[dim]not specified[/dim]")
    t.add_row("Save path", str(save) if save else "[dim]not specified[/dim]")
    console.print(t)

    # Progressive strategy
    theme.section("Progressive Strategy", console, "02")
    strat_table = theme.make_table()
    strat_table.add_column("Stage", style="bold cyan", no_wrap=True)
    strat_table.add_column("Cost", style="magenta")
    strat_table.add_column("Time")
    strat_table.add_column("Min Examples", style="dim")

    for step in OPTIMIZATION_STRATEGY:
        strat_table.add_row(
            step["name"],
            step["cost"],
            step["time"],
            str(step["min_examples"]),
        )
    console.print(strat_table)

    if trainset is None or valset is None:
        console.print()
        console.print(theme.warn(
            "Provide --trainset and --valset to run optimization. "
            "Requires MOSAICX_API_KEY."
        ))


# ---------------------------------------------------------------------------
# config (group)
# ---------------------------------------------------------------------------


@cli.group()
def config() -> None:
    """View and update MOSAICX configuration."""


@config.command("show")
def config_show() -> None:
    """Show current configuration."""
    cfg = get_config()
    dump = cfg.model_dump()

    theme.print_banner(_VERSION_NUMBER, console, lm=cfg.lm, lm_cheap=cfg.lm_cheap)

    # 01 · Language Models
    theme.section("Language Models", console, "01")
    t = theme.make_kv_table()
    t.add_row("lm", dump["lm"])
    t.add_row("lm_cheap", dump["lm_cheap"])
    t.add_row("api_base", dump["api_base"])
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
    """Set a configuration value."""
    console.print(theme.info(f"config set {key}={value}"))
    console.print(theme.warn(
        "Runtime config changes are not persisted yet. "
        "Use environment variables (MOSAICX_*) or a .env file."
    ))

# mosaicx/cli.py
"""
MOSAICX CLI v2 -- Click commands wired to real pipelines.

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

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .config import get_config

console = Console()

# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------

_VERSION = "mosaicx 2.0.0a1"


def _print_version(
    ctx: click.Context,
    _param: click.Parameter,
    value: bool,
) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(_VERSION)
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
    dspy.configure(lm=dspy.LM(cfg.lm, api_key=cfg.api_key))


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
# Main CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show version and exit.",
)
def cli() -> None:
    """MOSAICX -- Medical cOmputational Suite for Advanced Intelligent eXtraction."""


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
        console.print("[yellow]Warning: Low OCR quality detected. Results may be unreliable.[/yellow]")

    if doc.is_empty:
        raise click.ClickException(f"Document is empty: {document}")

    console.print(f"[dim]Loaded {doc.format} document ({doc.char_count} chars)[/dim]")

    # Resolve template to an output schema if not "auto"
    output_schema = None
    if template != "auto":
        template_path = Path(template)
        if template_path.exists() and template_path.suffix in (".yaml", ".yml"):
            from .schemas.template_compiler import compile_template_file

            try:
                output_schema = compile_template_file(template_path)
                console.print(f"[dim]Using template: {template_path.name}[/dim]")
            except Exception as exc:
                raise click.ClickException(f"Failed to compile template: {exc}")
        else:
            # Try registry lookup
            from .schemas.radreport.registry import get_template

            tpl_info = get_template(template)
            if tpl_info is not None:
                console.print(f"[dim]Using built-in template: {tpl_info.name} ({tpl_info.description})[/dim]")
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

    result = extractor(document_text=doc.text)

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

    console.print_json(json.dumps(output, indent=2, default=str))


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

    table = Table(title="Batch Configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("Input directory", str(input_dir))
    table.add_row("Output directory", str(output_dir))
    table.add_row("Template", template)
    table.add_row("Export formats", ", ".join(effective_formats))
    table.add_row("Workers", str(effective_workers))
    table.add_row("Completeness threshold", str(effective_threshold))
    table.add_row("Resume", str(resume))
    console.print(table)

    resume_id = "resume" if resume else None
    result = processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        process_fn=lambda doc: doc,  # placeholder until full wiring
        template=template,
        resume_id=resume_id,
    )
    console.print(f"[dim]Batch result: {result}[/dim]")


# ---------------------------------------------------------------------------
# template (group)
# ---------------------------------------------------------------------------


@cli.group()
def template() -> None:
    """Manage extraction templates."""


@template.command("create")
def template_create() -> None:
    """Create a new extraction template."""
    console.print("[bold]template create[/bold] -- not yet implemented")


@template.command("list")
def template_list() -> None:
    """List available templates."""
    from .schemas.radreport.registry import list_templates

    templates = list_templates()

    table = Table(title="Available Templates")
    table.add_column("Name", style="bold cyan")
    table.add_column("Exam Type")
    table.add_column("RadReport ID")
    table.add_column("Description")

    for tpl in templates:
        table.add_row(
            tpl.name,
            tpl.exam_type,
            tpl.radreport_id or "-",
            tpl.description,
        )

    console.print(table)


@template.command("validate")
@click.option("--file", "file_path", type=click.Path(exists=False, path_type=Path), required=True, help="Template YAML file to validate.")
def template_validate(file_path: Path) -> None:
    """Validate a template file."""
    from .schemas.template_compiler import compile_template_file

    if not file_path.exists():
        raise click.ClickException(f"File not found: {file_path}")

    try:
        model = compile_template_file(file_path)
        console.print(f"[bold green]Template is valid.[/bold green]")
        console.print(f"  Model name: {model.__name__}")
        console.print(f"  Fields: {list(model.model_fields.keys())}")
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
@click.option("--example-text", type=str, default="", help="Optional example document text for grounding.")
def schema_generate(description: str, example_text: str) -> None:
    """Generate a Pydantic schema from a description."""
    _configure_dspy()

    from .pipelines.schema_gen import SchemaGenerator

    generator = SchemaGenerator()
    result = generator(description=description, example_text=example_text)

    console.print("[bold green]Schema generated successfully.[/bold green]")
    console.print(f"  Model name: {result.compiled_model.__name__}")
    console.print(f"  Fields: {list(result.compiled_model.model_fields.keys())}")
    console.print("\n[bold]Schema spec:[/bold]")
    console.print_json(result.schema_spec.model_dump_json(indent=2))


@schema.command("list")
def schema_list() -> None:
    """List registered schemas."""
    console.print("[bold]schema list[/bold] -- not yet implemented")


@schema.command("refine")
def schema_refine() -> None:
    """Refine an existing schema."""
    console.print("[bold]schema refine[/bold] -- not yet implemented")


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
            console.print("[yellow]Warning: Low OCR quality detected. Results may be unreliable.[/yellow]")
        report_texts.append(doc.text)
    elif directory is not None:
        if not directory.is_dir():
            raise click.ClickException(f"Directory not found: {directory}")
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() in (".txt", ".md", ".markdown"):
                try:
                    doc = _load_doc_with_config(p)
                    if doc.quality_warning:
                        console.print("[yellow]Warning: Low OCR quality detected. Results may be unreliable.[/yellow]")
                    report_texts.append(doc.text)
                except Exception:
                    console.print(f"[yellow]Skipping {p.name}: unsupported or unreadable[/yellow]")
    else:
        raise click.ClickException("Provide --document or --dir.")

    if not report_texts:
        raise click.ClickException("No reports found to summarize.")

    console.print(f"[dim]Loaded {len(report_texts)} report(s)[/dim]")

    # Configure DSPy and run summarizer
    _configure_dspy()

    from .pipelines.summarizer import ReportSummarizer

    summarizer = ReportSummarizer()
    result = summarizer(reports=report_texts, patient_id=patient or "")

    console.print("\n[bold]Narrative Summary:[/bold]")
    console.print(result.narrative)

    if result.events:
        console.print(f"\n[dim]{len(result.events)} timeline event(s) extracted[/dim]")


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

    if regex_only:
        # Regex-only mode: no LLM needed
        for p in paths:
            text = p.read_text(encoding="utf-8")
            scrubbed = regex_scrub_phi(text)
            console.print(f"\n[bold]--- {p.name} ---[/bold]")
            console.print(scrubbed)
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
                console.print(f"[yellow]Skipping {p.name}: {exc}[/yellow]")
                continue
            if doc.quality_warning:
                console.print("[yellow]Warning: Low OCR quality detected. Results may be unreliable.[/yellow]")
            result = deid(document_text=doc.text, mode=mode)
            console.print(f"\n[bold]--- {p.name} ---[/bold]")
            console.print(result.redacted_text)


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

    table = Table(title="Optimization Configuration")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("Pipeline", pipeline or "(not specified)")
    table.add_row("Budget preset", budget)
    table.add_row("Strategy", opt_config.get("strategy", "N/A"))
    table.add_row("Max iterations", str(opt_config.get("max_iterations", "N/A")))
    table.add_row("Num candidates", str(opt_config.get("num_candidates", "N/A")))
    table.add_row("Training set", str(trainset) if trainset else "(not specified)")
    table.add_row("Validation set", str(valset) if valset else "(not specified)")
    table.add_row("Save path", str(save) if save else "(not specified)")
    console.print(table)

    # Display the progressive strategy info
    console.print("\n[bold]Progressive optimization strategy:[/bold]")
    for step in OPTIMIZATION_STRATEGY:
        console.print(
            f"  {step['name']}: cost {step['cost']}, "
            f"time {step['time']}, min examples {step['min_examples']}"
        )

    if trainset is None or valset is None:
        console.print(
            "\n[yellow]Note:[/yellow] Provide --trainset and --valset to run actual optimization. "
            "Optimization also requires a configured LLM (MOSAICX_API_KEY)."
        )


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

    table = Table(title="MOSAICX Configuration")
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value")

    for field_name, field_value in cfg.model_dump().items():
        # Mask the API key for security
        if field_name == "api_key" and field_value:
            display_value = field_value[:4] + "..." + field_value[-4:] if len(field_value) > 8 else "***"
        else:
            display_value = str(field_value)
        table.add_row(field_name, display_value)

    # Also show derived paths
    table.add_row("schema_dir", str(cfg.schema_dir))
    table.add_row("optimized_dir", str(cfg.optimized_dir))
    table.add_row("checkpoint_dir", str(cfg.checkpoint_dir))
    table.add_row("log_dir", str(cfg.log_dir))

    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value."""
    console.print(f"[bold]config set[/bold] {key}={value}")
    console.print("[yellow]Note:[/yellow] Runtime config changes are not persisted yet. "
                  "Use environment variables (MOSAICX_*) or a .env file.")

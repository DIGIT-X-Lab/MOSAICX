"""
MOSAICX CLI Application - Orchestrated Console Commands

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Structure first. Insight follows.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab
Department: Department of Radiology
University: LMU University Hospital | LMU Munich

Overview:
---------
Implements the public ``mosaicx`` Click command group, providing polished
terminal interactions for schema generation, PDF extraction, and longitudinal
summaries.  The module centralises shared styling, option handling, and
verbose feedback so the CLI mirrors the behaviour of the programmatic API.

Highlights:
-----------
- Rich-enabled help output with MOSAICX colour palette and typography.
- Schema lifecycle commands that integrate tightly with the registry module.
- Graceful fallbacks when optional dependencies (``rich_click``, Docling,
  Instructor, ReportLab) are absent.
- Shared ``styled_message`` helper usage for consistent status reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

try:
    import rich_click as click  # type: ignore[import-not-found]
    _rich_click_config = click.rich_click
except ModuleNotFoundError:  # pragma: no cover - fallback when rich_click missing
    import click  # type: ignore[no-redef]

    _rich_click_config = None
from rich.align import Align

from ..api import (
    extract_pdf,
    generate_schema,
    summarize_reports,
    standardize_document,
    save_standardize_artifacts,
)
from ..constants import (
    APPLICATION_NAME,
    APPLICATION_VERSION as __version__,
    AUTHOR_EMAIL as __email__,
    AUTHOR_NAME as __author__,
    DEFAULT_LLM_MODEL,
    MOSAICX_COLORS,
)
from ..display import console, show_main_banner, styled_message
from ..extractor import ExtractionError
from ..document_loader import DOC_SUFFIXES
from ..schema.registry import (
    cleanup_missing_files,
    get_schema_by_id,
    list_schemas,
    register_schema,
    scan_and_register_existing_schemas,
)
from ..summarizer import render_summary_rich, save_summary_artifacts
from ..utils import resolve_schema_reference
from ..utils.logging import setup_logging, get_current_log_file, get_logger, get_session_id

# Module-level logger
_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Click configuration (shared styling)
# ---------------------------------------------------------------------------

if _rich_click_config is not None:
    _rich_click_config.USE_RICH_MARKUP = True
    _rich_click_config.STYLE_OPTION = f"bold {MOSAICX_COLORS['primary']}"
    _rich_click_config.STYLE_ARGUMENT = f"bold {MOSAICX_COLORS['info']}"
    _rich_click_config.STYLE_COMMAND = f"bold {MOSAICX_COLORS['accent']}"
    _rich_click_config.STYLE_SWITCH = f"bold {MOSAICX_COLORS['success']}"
    _rich_click_config.STYLE_METAVAR = f"bold {MOSAICX_COLORS['warning']}"
    _rich_click_config.STYLE_USAGE = f"bold {MOSAICX_COLORS['primary']}"
    _rich_click_config.STYLE_USAGE_COMMAND = f"bold {MOSAICX_COLORS['accent']}"
    _rich_click_config.STYLE_HELPTEXT = f"{MOSAICX_COLORS['secondary']}"
    _rich_click_config.STYLE_HELPTEXT_FIRST_LINE = f"bold {MOSAICX_COLORS['secondary']}"
    _rich_click_config.STYLE_OPTION_DEFAULT = f"dim {MOSAICX_COLORS['muted']}"
    _rich_click_config.STYLE_REQUIRED_SHORT = f"bold {MOSAICX_COLORS['error']}"
    _rich_click_config.STYLE_REQUIRED_LONG = f"bold {MOSAICX_COLORS['error']}"

    _rich_click_config.USE_RICH_MARKUP = False
    _rich_click_config.USE_MARKDOWN = False
    _rich_click_config.SHOW_ARGUMENTS = True
    _rich_click_config.GROUP_ARGUMENTS_OPTIONS = True
    _rich_click_config.STYLE_OPTION = "dim"
    _rich_click_config.STYLE_ARGUMENT = "dim"
    _rich_click_config.STYLE_COMMAND = "bold"


def _deprecated_schema_output_alias(
    ctx: click.Context,
    param: click.Parameter,
    value: Optional[str],
) -> None:
    if not value:
        return
    value_path = Path(value)
    existing = ctx.params.get("schema_path")
    if existing and existing != value_path:
        raise click.BadParameter("Provide only one schema destination option.")
    ctx.params["schema_path"] = value_path
    alias = param.opts[0] if param.opts else param.name
    styled_message(f"{alias} is deprecated; use --schema-path instead.", "warning")


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name=APPLICATION_NAME)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default="DEBUG", help="Logging level for log file")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, log_level: str) -> None:
    """Primary MOSAICX CLI group."""
    
    # Initialize logging first
    log_file = setup_logging(level=log_level, console_output=verbose)
    session_id = get_session_id()
    _logger.info(f"MOSAICX CLI started (version {__version__})")
    
    if verbose:
        styled_message(f"📋 Session: {session_id} | Log: {log_file}", "info")

    show_main_banner()

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["log_file"] = log_file
    try:
        scan_and_register_existing_schemas()
    except Exception as exc:  # pragma: no cover - defensive guard
        _logger.warning(f"Schema auto-discovery skipped: {exc}")
        if verbose:
            styled_message(f"Schema auto-discovery skipped: {exc}", "warning")

    if ctx.invoked_subcommand is None:
        styled_message(
            "Welcome to MOSAICX! Use --help to see available commands.",
            "info",
        )


@cli.command()
@click.option("--desc", required=True, help="Natural language description of the data structure you want")
@click.option("--class-name", default="GeneratedModel", help="Name for the generated Pydantic class")
@click.option("--model", default=DEFAULT_LLM_MODEL, help="Model name for generation")
@click.option("--base-url", help="OpenAI-compatible API base URL")
@click.option("--api-key", help="API key for the endpoint")
@click.option("--temperature", type=float, default=0.2, help="Sampling temperature (0.0–2.0)")
@click.option("--schema-path", type=click.Path(path_type=Path), help="Write generated Pydantic schema (.py) to this path")
@click.option("--template", "store_as_template", is_flag=True, help="Save generated schema into the bundled template directory")
@click.option("--output", callback=_deprecated_schema_output_alias, expose_value=False, hidden=True)
@click.option("--save-model", callback=_deprecated_schema_output_alias, expose_value=False, hidden=True)
@click.option("--debug", is_flag=True, help="Verbose debug logs")
@click.pass_context
def generate(
    ctx: click.Context,
    desc: str,
    class_name: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
    schema_path: Optional[Path],
    store_as_template: bool,
    debug: bool,
) -> None:
    """Generate Pydantic schemas from natural language descriptions."""

    verbose = ctx.obj.get("verbose", False)

    if verbose:
        styled_message(f"Generating schema using model: {model}", "info")
        styled_message(f"Class name: {class_name}", "info")
        styled_message(f"Description: {desc}", "info")

    try:
        if store_as_template and schema_path is not None:
            raise click.BadParameter("Cannot use --template together with --schema-path.")

        with console.status(f"[{MOSAICX_COLORS['primary']}]Generating Pydantic model...", spinner="dots"):
            generated = generate_schema(
                description=desc,
                class_name=class_name,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
            )

        target_path = generated.write(schema_path, template=store_as_template)

        schema_id = register_schema(
            class_name=class_name,
            description=desc,
            file_path=target_path,
            model_used=model,
            temperature=temperature,
        )

        console.print()
        console.print()

        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table

        main_table = Table(
            title=f"✨ [bold {MOSAICX_COLORS['primary']}]Generated Schema Results[/bold {MOSAICX_COLORS['primary']}] ✨",
            title_style=f"bold {MOSAICX_COLORS['primary']}",
            border_style=MOSAICX_COLORS['accent'],
            show_header=True,
            header_style=f"bold {MOSAICX_COLORS['secondary']}",
            show_lines=True,
            expand=True,
            width=120,
            pad_edge=False,
        )

        main_table.add_column("Property", style=f"bold {MOSAICX_COLORS['secondary']}", width=18, justify="left")
        main_table.add_column("Details", style=MOSAICX_COLORS['primary'], width=100, justify="left")

        main_table.add_row("🏷️ Class Name", f"[{MOSAICX_COLORS['primary']}]{class_name}[/{MOSAICX_COLORS['primary']}]")
        main_table.add_row("🆔 Schema ID", f"[{MOSAICX_COLORS['primary']}]{schema_id}[/{MOSAICX_COLORS['primary']}]")
        main_table.add_row("📁 File Saved", f"[{MOSAICX_COLORS['primary']}]{target_path.name}[/{MOSAICX_COLORS['primary']}]")
        main_table.add_row("🤖 Model Used", f"[{MOSAICX_COLORS['primary']}]{model}[/{MOSAICX_COLORS['primary']}]")

        syntax = Syntax(
            generated.code,
            "python",
            theme="dracula",
            line_numbers=True,
            background_color="default",
            word_wrap=False,
        )

        code_panel = Panel(
            syntax,
            title="🐍 Generated Python Code",
            title_align="left",
            border_style=MOSAICX_COLORS['accent'],
            padding=(1, 2),
            width=96,
            expand=False,
        )

        main_table.add_row("💻 Code Preview", code_panel)
        console.print(Align.center(main_table))
        console.print()
        if store_as_template:
            styled_message(f"Stored template in: {target_path}", "success")
        elif schema_path:
            styled_message(f"Saved schema to: {target_path}", "success")

    except Exception as exc:
        styled_message(f"Schema generation failed: {exc}", "error")
        if debug:
            console.print_exception()
        raise click.ClickException(str(exc))


@cli.command("schemas")
@click.option("--class-name", help="Filter by class name (partial match)")
@click.option("--description", help="Filter by description (partial match)")
@click.option("--cleanup", is_flag=True, help="Remove entries for deleted files")
@click.option("--scan", is_flag=True, help="Scan and register existing untracked schema files")
@click.pass_context
def list_schemas_cmd(
    ctx: click.Context,
    class_name: Optional[str],
    description: Optional[str],
    cleanup: bool,
    scan: bool,
) -> None:
    """List all generated schemas with details."""

    if cleanup:
        removed_count = cleanup_missing_files()
        if removed_count > 0:
            styled_message(f"Removed {removed_count} entries for missing files", "success")
        else:
            styled_message("No missing files found", "info")

    if scan:
        styled_message("Scanning for existing schema files...", "info")
        registered_count = scan_and_register_existing_schemas()
        if registered_count > 0:
            styled_message(f"Registered {registered_count} existing schema files", "success")
        else:
            styled_message("No new schema files found to register", "info")

    schemas = list_schemas(class_name_filter=class_name, description_filter=description)

    if not schemas:
        styled_message("No schemas found. Generate some schemas first!", "warning")
        return

    console.print()
    console.print()

    from rich.columns import Columns
    from rich.panel import Panel

    schema_cards = []
    for schema in schemas:
        status = "✅ Exists" if schema["file_exists"] else "❌ Missing"
        status_color = MOSAICX_COLORS["success"] if schema["file_exists"] else MOSAICX_COLORS["error"]

        created_at = schema["created_at"]
        if "T" in created_at:
            date_part = created_at.split("T")[0]
            time_part = created_at.split("T")[1][:8]
            formatted_date = f"{date_part} {time_part}"
        else:
            formatted_date = created_at[:19]

        card_content = (
            f"[bold {MOSAICX_COLORS['primary']}]Schema ID:[/bold {MOSAICX_COLORS['primary']}] {schema['id']}\n"
            f"[bold {MOSAICX_COLORS['accent']}]Class:[/bold {MOSAICX_COLORS['accent']}] {schema['class_name']}\n"
            f"[bold {MOSAICX_COLORS['secondary']}]Description:[/bold {MOSAICX_COLORS['secondary']}] {schema['description']}\n\n"
            f"[bold {MOSAICX_COLORS['info']}]File:[/bold {MOSAICX_COLORS['info']}] {schema['file_name']}\n"
            f"[bold {MOSAICX_COLORS['muted']}]Model:[/bold {MOSAICX_COLORS['muted']}] {schema['model_used']}\n"
            f"[bold {MOSAICX_COLORS['muted']}]Created:[/bold {MOSAICX_COLORS['muted']}] {formatted_date}\n"
            f"[bold]Status:[/bold] [{status_color}]{status}[/{status_color}]"
        )

        schema_panel = Panel(
            card_content,
            title=f"[bold {MOSAICX_COLORS['primary']}]{schema['class_name']}[/bold {MOSAICX_COLORS['primary']}]",
            title_align="left",
            border_style=MOSAICX_COLORS['accent'],
            padding=(1, 2),
            width=60,
        )
        schema_cards.append(schema_panel)

    console.print(
        f"\n[bold {MOSAICX_COLORS['primary']}]📚 Generated Schemas Registry[/bold {MOSAICX_COLORS['primary']}]",
        justify="center",
    )
    console.print(
        f"[{MOSAICX_COLORS['muted']}]Found {len(schemas)} schema(s)[/{MOSAICX_COLORS['muted']}]",
        justify="center",
    )
    console.print()

    for i in range(0, len(schema_cards), 2):
        row_cards = schema_cards[i : i + 2]
        console.print(Columns(row_cards, equal=True, expand=True))
        console.print()

    console.print()
    styled_message(
        "💡 Tip: Use schema ID, filename, or file path in extract commands",
        "info",
    )


@cli.command()
@click.option(
    "--document",
    "--doc",
    "document",
    required=True,
    type=click.Path(exists=True),
    help="Path to a clinical document (PDF, DOCX, PPTX, TXT, ...)",
)
@click.option("--schema", required=True, help="Schema identifier (ID, filename, or file path)")
@click.option("--model", default=DEFAULT_LLM_MODEL, help="Model name for extraction")
@click.option("--base-url", help="OpenAI-compatible API base URL")
@click.option("--api-key", help="API key for the endpoint")
@click.option(
    "--backend",
    type=click.Choice(["auto", "ollama", "vllm", "openai", "anthropic", "sglang", "tgi", "llamacpp"], case_sensitive=False),
    default="auto",
    help="LLM backend for Outlines structured generation (auto-detected from URL if not specified)",
)
@click.option("--temperature", type=float, default=0.0, show_default=True, help="Sampling temperature (0.0–2.0)")
@click.option("--output", "--save", "output_path", type=click.Path(path_type=Path), help="Save extracted JSON result to this path")
@click.option("--debug", is_flag=True, help="Verbose debug logs")
@click.pass_context
def extract(
    ctx: click.Context,
    document: str,
    schema: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    backend: str,
    temperature: float,
    output_path: Optional[Path],
    debug: bool,
) -> None:
    """Extract structured data from a clinical document using a Pydantic schema."""

    verbose = ctx.obj.get("verbose", False)
    log_file = ctx.obj.get("log_file")
    
    _logger.info(f"Extract command invoked: document={document}, schema={schema}, model={model}")

    if verbose:
        styled_message(f"Extracting from: {document}", "info")
        styled_message(f"Using schema: {schema}", "info")
        styled_message(f"Using model: {model}", "info")
        if base_url:
            styled_message(f"Base URL: {base_url}", "info")
        if log_file:
            styled_message(f"📋 Full logs: {log_file}", "info")

    try:
        with console.status(
            f"[{MOSAICX_COLORS['primary']}]Preparing extraction...", spinner="dots"
        ) as status:
            status.update(f"[{MOSAICX_COLORS['accent']}]Resolving schema reference…")
            resolved_schema_path = resolve_schema_reference(schema)
            if not resolved_schema_path:
                _logger.error(f"Schema not found: {schema}")
                raise click.ClickException(f"Could not find schema: {schema}")

            _logger.info(f"Resolved schema path: {resolved_schema_path}")
            if verbose:
                styled_message(f"Resolved schema to: {resolved_schema_path}", "info")

            status.update(f"[{MOSAICX_COLORS['accent']}]Loading schema metadata…")
            all_schemas = list_schemas()
            schema_class_name = None
            resolved_path_str = str(resolved_schema_path)
            for schema_info in all_schemas:
                if resolved_path_str == schema_info["file_path"]:
                    schema_class_name = schema_info["class_name"]
                    break

            if schema_class_name is None:
                try:
                    from ..extractor import load_schema_model

                    schema_class = load_schema_model(str(resolved_schema_path))
                    schema_class_name = schema_class.__name__
                except Exception:
                    schema_class_name = resolved_schema_path.stem

            _logger.info(f"Using schema class: {schema_class_name}")
            if verbose:
                styled_message(f"Using schema class: {schema_class_name}", "info")
                if backend != "auto":
                    styled_message(f"Using backend: {backend}", "info")

            running_message = (
                f"[{MOSAICX_COLORS['accent']}]Running extraction with {model}…"
            )
            status.update(running_message)
            fallback_used = {"flag": False}

            def _report_fallback(message: str) -> None:
                fallback_used["flag"] = True
                _logger.info(f"Fallback triggered: {message}")
                status.update(f"[{MOSAICX_COLORS['accent']}]Using {message}…")

            extraction = extract_pdf(
                pdf_path=document,
                schema_path=resolved_schema_path,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                backend=backend,
                status_callback=_report_fallback,
            )
            if fallback_used["flag"]:
                status.update(running_message)

        console.print()
        result_dict = extraction.record.model_dump()
        _logger.info(f"Extraction completed successfully for {document}")

        console.print()
        styled_message(f"📋 Extraction results based on schema: {schema}", "primary", center=True)
        console.print()

        from rich.table import Table

        data_table = Table(
            show_lines=False,
            border_style=MOSAICX_COLORS["secondary"],
            header_style=f"bold {MOSAICX_COLORS['primary']}",
        )

        data_table.add_column("Field", style=MOSAICX_COLORS["info"], no_wrap=True)
        data_table.add_column("Extracted Value", style=MOSAICX_COLORS["accent"])

        for field_name, value in result_dict.items():
            if value is None:
                display_value = "[dim]Not found[/dim]"
            elif isinstance(value, (list, dict)):
                text = str(value)
                display_value = text[:50] + "..." if len(text) > 50 else text
            else:
                display_value = str(value)

            data_table.add_row(field_name, display_value)

        console.print(Align.center(data_table))

        if output_path:
            extraction.write_json(output_path)
            console.print()
            console.print()
            styled_message("📁 EXTRACTION SAVED", "accent", center=True)
            console.print()
            styled_message(f"JSON: {output_path.name}", "primary", center=True)

        if verbose and debug:
            console.print()
            styled_message("Raw extracted data:", "secondary", center=True)
            console.print()
            from rich.json import JSON

            console.print(JSON(extraction.record.model_dump_json(indent=2)))

    except ExtractionError as exc:
        _logger.error(f"Extraction failed: {exc}", exc_info=True)
        styled_message(f"Extraction failed: {exc}", "error")
        if log_file:
            styled_message(f"📋 See full log: {log_file}", "info")
        if debug:
            console.print_exception()
        raise click.ClickException(str(exc))
    except Exception as exc:
        _logger.error(f"Unexpected error: {exc}", exc_info=True)
        styled_message(f"Unexpected error: {exc}", "error")
        if log_file:
            styled_message(f"📋 See full log: {log_file}", "info")
        if debug:
            console.print_exception()
        raise click.ClickException(str(exc))


@cli.command("log-path")
def show_log_path() -> None:
    """Show the path to the MOSAICX log file and session info."""
    session_id = get_session_id()
    log_file = get_current_log_file()
    if log_file:
        styled_message(f"📋 Session: {session_id}", "info")
        styled_message(f"   Log file: {log_file}", "info")
        if log_file.exists():
            size_kb = log_file.stat().st_size / 1024
            styled_message(f"   Size: {size_kb:.1f} KB", "secondary")
    else:
        styled_message("Logging not initialized", "warning")


@cli.command()
@click.option(
    "--document",
    "--doc",
    "document",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the clinical document to standardize.",
)
@click.option("--schema", required=True, help="Schema identifier (ID, filename, or file path).")
@click.option("--model", default=DEFAULT_LLM_MODEL, show_default=True, help="Model used for structuring.")
@click.option("--base-url", help="OpenAI-compatible API base URL.")
@click.option("--api-key", help="API key for the endpoint.")
@click.option("--temperature", type=float, default=0.0, show_default=True, help="Sampling temperature for the LLM.")
@click.option(
    "--artifact",
    "artifacts",
    type=click.Choice(["json", "pdf", "txt", "docx"], case_sensitive=False),
    multiple=True,
    default=("json",),
    help="Choose one or more output artifacts (repeat flag for multiples).",
)
@click.option(
    "--narrative/--no-narrative",
    default=False,
    help="Generate an LLM-written narrative summary to accompany the structured output.",
)
@click.option(
    "--narrative-temperature",
    type=float,
    default=None,
    help="Sampling temperature for the narrative LLM (defaults to 0.2 when not provided).",
)
@click.option("--out-dir", type=click.Path(path_type=Path), help="Directory for auto-named artifacts.")
@click.option("--json-out", type=click.Path(path_type=Path), help="Explicit path for JSON output.")
@click.option("--pdf-out", type=click.Path(path_type=Path), help="Explicit path for PDF output.")
@click.option("--text-out", type=click.Path(path_type=Path), help="Explicit path for TXT output.")
@click.option("--docx-out", type=click.Path(path_type=Path), help="Explicit path for DOCX output.")
@click.option(
    "--record-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Use a pre-extracted JSON record (skips document extraction).",
)
@click.option("--debug", is_flag=True, help="Show tracebacks on errors.")
@click.pass_context
def standardize(
    ctx: click.Context,
    document: Optional[Path],
    schema: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
    artifacts: tuple[str, ...],
    narrative: bool,
    narrative_temperature: Optional[float],
    out_dir: Optional[Path],
    json_out: Optional[Path],
    pdf_out: Optional[Path],
    text_out: Optional[Path],
    docx_out: Optional[Path],
    record_json: Optional[Path],
    debug: bool,
) -> None:
    """Standardize a document into a structured report using a schema."""

    verbose = ctx.obj.get("verbose", False)

    record_payload: Optional[dict] = None
    if record_json is not None:
        try:
            record_payload = json.loads(record_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in {record_json}: {exc}") from exc
        if verbose:
            styled_message(f"Using pre-extracted JSON: {record_json}", "info")

    if document is None and record_payload is None:
        raise click.ClickException("Provide either --document or --record-json (or both).")

    if verbose and document is not None:
        styled_message(f"Standardizing document: {document}", "info")
    if verbose:
        styled_message(f"Using schema: {schema}", "info")
        styled_message(f"Model: {model}", "info")

    try:
        with console.status(
            f"[{MOSAICX_COLORS['primary']}]Preparing standardization...", spinner="dots"
        ) as status:
            status.update(f"[{MOSAICX_COLORS['accent']}]Resolving schema reference…")
            resolved_schema_path = resolve_schema_reference(schema)
            if not resolved_schema_path:
                raise click.ClickException(f"Could not find schema: {schema}")

            status.update(f"[{MOSAICX_COLORS['accent']}]Loading schema module…")
            fallback_used = {"flag": False}

            def _report(message: str) -> None:
                fallback_used["flag"] = True
                status.update(f"[{MOSAICX_COLORS['accent']}]Using {message}…")

            result = standardize_document(
                document_path=document,
                schema_path=resolved_schema_path,
                record_data=record_payload,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                generate_narrative=narrative,
                narrative_temperature=narrative_temperature,
                status_callback=_report,
            )
            if fallback_used["flag"]:
                status.update(f"[{MOSAICX_COLORS['accent']}]Finalizing structured report…")

        console.print()
        styled_message(
            f"📊 Structured report ({result.schema_name})", "primary", center=True
        )
        console.print()

        if result.narrative:
            from rich.panel import Panel

            console.print(
                Panel(
                    result.narrative,
                    title="Narrative Summary",
                    border_style=MOSAICX_COLORS["accent"],
                )
            )
            console.print()
        elif narrative and result.narrative_error:
            msg = result.narrative_error.splitlines()[0]
            if len(msg) > 160:
                msg = msg[:157] + "..."
            styled_message(f"Narrative generation failed: {msg}", "warning")
            console.print()

        from rich.table import Table

        table = Table(
            show_lines=False,
            border_style=MOSAICX_COLORS["secondary"],
            header_style=f"bold {MOSAICX_COLORS['primary']}",
        )
        table.add_column("Field", style=MOSAICX_COLORS["info"], no_wrap=True)
        table.add_column("Value", style=MOSAICX_COLORS["accent"])

        for field, value in result.to_dict().items():
            if value is None:
                display_value = "[dim]Not provided[/dim]"
            elif isinstance(value, (list, dict)):
                text = json.dumps(value, ensure_ascii=False)
                display_value = text[:70] + "…" if len(text) > 70 else text
            else:
                display_value = str(value)
            table.add_row(field, display_value)

        console.print(Align.center(table))

        normalized_artifacts = tuple(dict.fromkeys(a.lower() for a in artifacts)) or ("json",)
        selected = set(normalized_artifacts)

        effective_json = json_out
        if effective_json and "json" not in selected:
            styled_message("Ignoring --json-out because JSON artifact disabled.", "warning")
            effective_json = None

        wants_text = any(opt in selected for opt in ("txt", "text"))
        effective_text = text_out
        if effective_text and not wants_text:
            styled_message("Ignoring --text-out because TXT artifact disabled.", "warning")
            effective_text = None

        effective_pdf = pdf_out
        if effective_pdf and "pdf" not in selected:
            styled_message("Ignoring --pdf-out because PDF artifact disabled.", "warning")
            effective_pdf = None

        effective_docx = docx_out
        if effective_docx and "docx" not in selected:
            styled_message("Ignoring --docx-out because DOCX artifact disabled.", "warning")
            effective_docx = None

        save_standardize_artifacts(
            result,
            artifacts=normalized_artifacts,
            json_path=effective_json,
            pdf_path=effective_pdf,
            text_path=effective_text,
            docx_path=effective_docx,
            out_dir=out_dir,
            emit_messages=True,
        )

    except ExtractionError as exc:
        styled_message(f"Standardization failed: {exc}", "error")
        if debug:
            console.print_exception()
        raise click.ClickException(str(exc))
    except Exception as exc:
        styled_message(f"Unexpected error: {exc}", "error")
        if debug:
            console.print_exception()
        raise click.ClickException(str(exc))


@cli.command()
@click.option(
    "--report",
    "reports",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="One or more report files (PDF, DOCX, PPTX, TXT, ...). Use multiple --report flags.",
)
@click.option(
    "--dir",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing reports (recursively scans supported formats).",
)
@click.option("--patient", "patient_id", type=str, help="Patient identifier (optional).")
@click.option("--model", default=DEFAULT_LLM_MODEL, show_default=True, help="Model for summarization.")
@click.option("--base-url", help="OpenAI-compatible API base URL (e.g., http://localhost:11434/v1).")
@click.option("--api-key", help="API key (e.g., 'ollama' for Ollama).")
@click.option("--temperature", type=float, default=0.2, show_default=True, help="Sampling temperature.")
@click.option("--output", "--json-out", "json_out", type=click.Path(path_type=Path), help="Save a JSON summary to this path.")
@click.option(
    "--format",
    "formats",
    type=click.Choice(["json", "pdf"], case_sensitive=False),
    multiple=True,
    default=("json",),
    help="Choose output artifact(s); repeat for multiple values (default: json).",
)
@click.option("--pdf-out", type=click.Path(path_type=Path), help="Save a PDF summary to this path.")
@click.option("--debug", is_flag=True, help="Show tracebacks on errors.")
@click.pass_context
def summarize(
    ctx: click.Context,
    reports: tuple[Path, ...],
    input_dir: Optional[Path],
    patient_id: Optional[str],
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
    json_out: Optional[Path],
    formats: tuple[str, ...],
    pdf_out: Optional[Path],
    debug: bool,
) -> None:
    """Summarize one or many reports (same patient) into a critical timeline and concise overall summary."""

    verbose = ctx.obj.get("verbose", False)

    allowed_suffixes = set(DOC_SUFFIXES.keys())

    def _supported(path: Path) -> bool:
        return path.suffix.lower() in allowed_suffixes

    paths: List[Path] = list(reports)
    invalid_reports = [p for p in paths if not _supported(p)]
    if invalid_reports:
        suffixes = ", ".join(sorted({p.suffix.lower() or "<none>" for p in invalid_reports}))
        raise click.ClickException(
            f"Unsupported report extension(s): {suffixes}. "
            f"Supported formats: {', '.join(sorted(allowed_suffixes))}."
        )

    if input_dir:
        for candidate in input_dir.rglob("*"):
            if _supported(candidate):
                paths.append(candidate)

    if not paths:
        raise click.ClickException(
            "Provide at least one --report or a --dir containing supported formats "
            f"({', '.join(sorted(allowed_suffixes))})."
        )

    if verbose:
        styled_message(f"Summarizing {len(paths)} file(s)", "info")
        if patient_id:
            styled_message(f"Patient: {patient_id}", "info")

    try:
        with console.status(f"[{MOSAICX_COLORS['primary']}]Summarizing reports...", spinner="dots"):
            summary = summarize_reports(
                paths,
                patient_id=patient_id,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
            )

        render_summary_rich(summary)

        normalized_formats = tuple(dict.fromkeys(f.lower() for f in formats)) if formats else ("json",)
        if not normalized_formats:
            normalized_formats = ("json",)
        selected_formats = set(normalized_formats)

        json_path = json_out
        if json_path and "json" not in selected_formats:
            styled_message("Ignoring --output because JSON artifact disabled.", "warning")
            json_path = None

        pdf_path = pdf_out
        if pdf_path and "pdf" not in selected_formats:
            styled_message("Ignoring --pdf-out because PDF artifact disabled.", "warning")
            pdf_path = None

        save_summary_artifacts(
            summary,
            artifacts=normalized_formats,
            json_path=json_path,
            pdf_path=pdf_path,
            patient_id=patient_id,
            model_name=model,
            emit_messages=True,
        )

    except Exception as exc:
        styled_message(f"Summarization failed: {exc}", "error")
        if debug:
            console.print_exception()
        raise click.ClickException(str(exc))


def main(args: Optional[List[str]] = None) -> None:
    """Execute the MOSAICX CLI."""

    cli(args)


__all__ = [
    "cli",
    "generate",
    "list_schemas_cmd",
    "extract",
    "standardize",
    "summarize",
    "main",
]

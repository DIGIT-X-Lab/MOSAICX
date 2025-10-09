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

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

try:
    import rich_click as click  # type: ignore[import-not-found]
    _rich_click_config = click.rich_click
except ModuleNotFoundError:  # pragma: no cover - fallback when rich_click missing
    import click  # type: ignore[no-redef]

    _rich_click_config = None
from rich.align import Align

from ..api import extract_pdf, generate_schema, summarize_reports
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
from ..schema.registry import (
    cleanup_missing_files,
    get_schema_by_id,
    list_schemas,
    register_schema,
    scan_and_register_existing_schemas,
)
from ..summarizer import render_summary_rich, write_summary_json
from ..utils import resolve_schema_reference


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
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Primary MOSAICX CLI group."""

    show_main_banner()

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    try:
        scan_and_register_existing_schemas()
    except Exception as exc:  # pragma: no cover - defensive guard
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
@click.option("--pdf", required=True, type=click.Path(exists=True), help="Path to PDF file to extract from")
@click.option("--schema", required=True, help="Schema identifier (ID, filename, or file path)")
@click.option("--model", default=DEFAULT_LLM_MODEL, help="Model name for extraction")
@click.option("--base-url", help="OpenAI-compatible API base URL")
@click.option("--api-key", help="API key for the endpoint")
@click.option("--temperature", type=float, default=0.0, show_default=True, help="Sampling temperature (0.0–2.0)")
@click.option("--output", "--save", "output_path", type=click.Path(path_type=Path), help="Save extracted JSON result to this path")
@click.option("--debug", is_flag=True, help="Verbose debug logs")
@click.pass_context
def extract(
    ctx: click.Context,
    pdf: str,
    schema: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
    output_path: Optional[Path],
    debug: bool,
) -> None:
    """Extract structured data from PDF using a generated Pydantic schema."""

    verbose = ctx.obj.get("verbose", False)

    if verbose:
        styled_message(f"Extracting from: {pdf}", "info")
        styled_message(f"Using schema: {schema}", "info")
        styled_message(f"Using model: {model}", "info")
        if base_url:
            styled_message(f"Base URL: {base_url}", "info")

    try:
        with console.status(
            f"[{MOSAICX_COLORS['primary']}]Preparing extraction...", spinner="dots"
        ) as status:
            status.update(f"[{MOSAICX_COLORS['accent']}]Resolving schema reference…")
            resolved_schema_path = resolve_schema_reference(schema)
            if not resolved_schema_path:
                raise click.ClickException(f"Could not find schema: {schema}")

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

            if verbose:
                styled_message(f"Using schema class: {schema_class_name}", "info")

            status.update(
                f"[{MOSAICX_COLORS['accent']}]Running extraction with {model}…"
            )
            extraction = extract_pdf(
                pdf_path=pdf,
                schema_path=resolved_schema_path,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
            )

        console.print()
        result_dict = extraction.record.model_dump()

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
        styled_message(f"Extraction failed: {exc}", "error")
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
    help="One or more report files (.pdf or .txt). Use multiple --report flags.",
)
@click.option(
    "--dir",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing reports (recursively scans *.pdf, *.txt).",
)
@click.option("--patient", "patient_id", type=str, help="Patient identifier (optional).")
@click.option("--model", default=DEFAULT_LLM_MODEL, show_default=True, help="Model for summarization.")
@click.option("--base-url", help="OpenAI-compatible API base URL (e.g., http://localhost:11434/v1).")
@click.option("--api-key", help="API key (e.g., 'ollama' for Ollama).")
@click.option("--temperature", type=float, default=0.2, show_default=True, help="Sampling temperature.")
@click.option("--output", "--json-out", "json_out", type=click.Path(path_type=Path), help="Save a JSON summary to this path.")
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
    debug: bool,
) -> None:
    """Summarize one or many reports (same patient) into a critical timeline and concise overall summary."""

    verbose = ctx.obj.get("verbose", False)

    paths: List[Path] = list(reports)
    if input_dir:
        for candidate in input_dir.rglob("*"):
            if candidate.suffix.lower() in {".pdf", ".txt"}:
                paths.append(candidate)

    if not paths:
        raise click.ClickException("Provide at least one --report or a --dir with .pdf/.txt files.")

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

        if json_out is None:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            base = (patient_id or "patient").lower()
            json_out = Path("output") / f"summary_{base}_{ts}.json"

        write_summary_json(summary, json_out)
        styled_message(f"Saved JSON: {json_out}", "accent")

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
    "summarize",
    "main",
]

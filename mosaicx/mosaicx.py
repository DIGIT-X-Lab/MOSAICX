"""
MOSAICX Main Module - Application Entry Point and Core Functionality

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Overview:
---------
This module serves as the main entry point for the MOSAICX application, providing
a comprehensive command-line interface for medical data extraction and processing.
It orchestrates the various components of the system including schema generation,
natural language processing, and data validation using the schema.builder module
as the core engine.

Core Functionality:
------------------
• Main CLI command group with rich-click integration
• Application banner and branding display
• Schema generation from natural language descriptions
• PDF extraction with structured data output
• Integration with Ollama for local LLM processing
• Pydantic model compilation and code generation

Architecture:
------------
Built using Click framework with rich-click enhancements for modern CLI UX.
Uses schema.builder module as the core working prototype for all schema operations.

Usage Examples:
--------------
Generate schema from natural language:
    >>> mosaicx generate --desc "Patient demographics with age, gender"
    >>> mosaicx generate --desc "Blood test results" --model llama3

Extract data from PDF:
    >>> mosaicx extract --pdf report.pdf --schema PatientRecord

Show banner:
    >>> mosaicx banner

Dependencies:
------------
External Libraries:
    • rich-click (^1.0.0): Enhanced command-line interface framework
    • schema.builder: Core schema generation engine (working prototype)
    • extractor: PDF processing and data extraction engine

Module Metadata:
---------------
Author:        Lalith Kumar Shiyam Sundar, PhD
Email:         Lalith.shiyam@med.uni-muenchen.de  
Institution:   DIGIT-X Lab, LMU Radiology | LMU University Hospital
License:       AGPL-3.0 (GNU Affero General Public License v3.0)
Version:       1.0.0
Created:       2025-09-18
Last Modified: 2025-09-18

Copyright Notice:
----------------
© 2025 DIGIT-X Lab, LMU Radiology | LMU University Hospital
This software is distributed under the AGPL-3.0 license.
See LICENSE file for full terms and conditions.
"""

from typing import List, Optional
from pathlib import Path
import rich_click as click

from .display import show_main_banner, console, styled_message
from rich.align import Align
from .schema.builder import synthesize_pydantic_model
from .extractor import extract_from_pdf, ExtractionError
from .schema.registry import (
    register_schema, 
    list_schemas, 
    get_schema_by_id, 
    get_suggested_filename,
    cleanup_missing_files,
    scan_and_register_existing_schemas
)

# Import metadata from constants
from .constants import (
    APPLICATION_NAME,
    APPLICATION_VERSION as __version__,
    AUTHOR_NAME as __author__,
    AUTHOR_EMAIL as __email__,
    DEFAULT_LLM_MODEL,
    MOSAICX_COLORS,
    PACKAGE_SCHEMA_PYD_DIR
)

# Configure rich-click with Dracula theme colors
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_OPTION = f"bold {MOSAICX_COLORS['primary']}"
click.rich_click.STYLE_ARGUMENT = f"bold {MOSAICX_COLORS['info']}"
click.rich_click.STYLE_COMMAND = f"bold {MOSAICX_COLORS['accent']}"
click.rich_click.STYLE_SWITCH = f"bold {MOSAICX_COLORS['success']}"
click.rich_click.STYLE_METAVAR = f"bold {MOSAICX_COLORS['warning']}"
click.rich_click.STYLE_USAGE = f"bold {MOSAICX_COLORS['primary']}"
click.rich_click.STYLE_USAGE_COMMAND = f"bold {MOSAICX_COLORS['accent']}"
click.rich_click.STYLE_HELPTEXT = f"{MOSAICX_COLORS['secondary']}"
click.rich_click.STYLE_HELPTEXT_FIRST_LINE = f"bold {MOSAICX_COLORS['secondary']}"
click.rich_click.STYLE_OPTION_DEFAULT = f"dim {MOSAICX_COLORS['muted']}"
click.rich_click.STYLE_REQUIRED_SHORT = f"bold {MOSAICX_COLORS['error']}"
click.rich_click.STYLE_REQUIRED_LONG = f"bold {MOSAICX_COLORS['error']}"

# Configure rich-click for professional CLI appearance
click.rich_click.USE_RICH_MARKUP = False  # Disable colorful markup
click.rich_click.USE_MARKDOWN = False     # Disable markdown formatting
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_OPTION = "dim"
click.rich_click.STYLE_ARGUMENT = "dim"
click.rich_click.STYLE_COMMAND = "bold"


def _resolve_schema_reference(schema_ref: str) -> Optional[Path]:
    """Resolve a schema reference to an actual file path.
    
    Args:
        schema_ref: Can be a Schema ID, filename, or file path
        
    Returns:
        Path object pointing to the schema file, or None if not found
    """
    from pathlib import Path
    
    # Try as Schema ID first
    schema_by_id = get_schema_by_id(schema_ref)
    if schema_by_id:
        schema_path = Path(schema_by_id['file_path'])
        if schema_path.exists():
            return schema_path
    
    # Try as filename in the schema directory
    schema_dir = Path(PACKAGE_SCHEMA_PYD_DIR)
    if schema_dir.exists():
        # Direct filename match
        filename_path = schema_dir / schema_ref
        if filename_path.exists() and filename_path.suffix == '.py':
            return filename_path
        
        # Add .py extension if missing
        if not schema_ref.endswith('.py'):
            filename_with_ext = schema_dir / f"{schema_ref}.py"
            if filename_with_ext.exists():
                return filename_with_ext
    
    # Try as direct file path
    direct_path = Path(schema_ref)
    if direct_path.exists() and direct_path.suffix == '.py':
        return direct_path
    
    # Try relative to current directory
    if not direct_path.is_absolute():
        current_dir_path = Path.cwd() / schema_ref
        if current_dir_path.exists() and current_dir_path.suffix == '.py':
            return current_dir_path
    
    return None


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name=APPLICATION_NAME)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    **MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction**
    
    LLMS for Intelligent Structuring • Summarization • Classification
    
    Transform unstructured medical reports into validated, structured data schemas
    using local LLM processing and advanced natural language understanding.
    """
    # Always show banner first
    show_main_banner()
    
    # Store verbose flag in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # If no subcommand provided, show welcome message
    if ctx.invoked_subcommand is None:
        styled_message(
            "Welcome to MOSAICX! Use --help to see available commands.",
            "info"
        )


@cli.command()
@click.option("--desc", required=True, help="Natural language description of the data structure you want")
@click.option("--class-name", default="GeneratedModel", help="Name for the generated Pydantic class")
@click.option("--model", default=DEFAULT_LLM_MODEL, help="Model name for generation")
@click.option("--base-url", help="OpenAI-compatible API base URL")
@click.option("--api-key", help="API key for the endpoint")
@click.option("--temperature", type=float, default=0.2, help="Sampling temperature (0.0–2.0)")
@click.option("--save-model", type=click.Path(), help="Write generated Pydantic class .py to this path")
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
    save_model: Optional[str], 
    debug: bool
) -> None:
    """Generate Pydantic schemas from natural language descriptions.
    
    MODEL COMPATIBILITY:
    ⭐ gpt-oss:120b (default) | 🟢 mistral:latest | 🟢 qwen2.5-coder:32b | 🟢 deepseek-r1:8b
    """
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        styled_message(f"Generating schema using model: {model}", "info")
        styled_message(f"Class name: {class_name}", "info")
        styled_message(f"Description: {desc}", "info")
    
    try:
        # Use the new simplified schema builder
        with console.status(f"[{MOSAICX_COLORS['primary']}]Generating Pydantic model...", spinner="dots"):
            class_code = synthesize_pydantic_model(
                description=desc,
                class_name=class_name,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature
            )
        
        # Auto-generate better filename with description context
        suggested_filename = get_suggested_filename(class_name, desc)
        
        # Determine save path (use suggested filename if not specified)
        py_save_path = save_model if save_model else Path(PACKAGE_SCHEMA_PYD_DIR) / suggested_filename
        
        # Ensure directory exists
        Path(py_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save Python code
        Path(py_save_path).write_text(class_code)
        
        # Register the schema in the registry for easy management
        schema_id = register_schema(
            class_name=class_name,
            description=desc,
            file_path=Path(py_save_path),
            model_used=model,
            temperature=temperature
        )
        
        # Create a beautiful comprehensive table with all the information
        console.print()
        console.print()
        
        from rich.table import Table
        from rich.syntax import Syntax
        from rich.panel import Panel
        
        # Main table with nested information
        main_table = Table(
            title=f"✨ [bold {MOSAICX_COLORS['primary']}]Generated Schema Results[/bold {MOSAICX_COLORS['primary']}] ✨",
            title_style=f"bold {MOSAICX_COLORS['primary']}",
            border_style=MOSAICX_COLORS['accent'],
            show_header=True,
            header_style=f"bold {MOSAICX_COLORS['secondary']}",
            show_lines=True,
            expand=True,
            width=120,
            pad_edge=False
        )
        
        main_table.add_column("Property", style=f"bold {MOSAICX_COLORS['secondary']}", width=18, justify="left")
        main_table.add_column("Details", style=MOSAICX_COLORS['primary'], width=100, justify="left")
        
        # Add rows with all the information - ensuring consistent formatting
        main_table.add_row(
            "🏷️ Class Name", 
            f"[bold {MOSAICX_COLORS['primary']}]{class_name}[/bold {MOSAICX_COLORS['primary']}]"
        )
        main_table.add_row(
            "🆔 Schema ID", 
            f"[{MOSAICX_COLORS['primary']}]{schema_id}[/{MOSAICX_COLORS['primary']}]"
        )
        main_table.add_row(
            "📁 File Saved", 
            f"[{MOSAICX_COLORS['primary']}]{Path(py_save_path).name}[/{MOSAICX_COLORS['primary']}]"
        )
        main_table.add_row(
            "🤖 Model Used", 
            f"[{MOSAICX_COLORS['primary']}]{model}[/{MOSAICX_COLORS['primary']}]"
        )
        
        # Create syntax highlighted code in a panel with full width
        syntax = Syntax(
            class_code, 
            "python", 
            theme="dracula", 
            line_numbers=True,
            background_color="default",
            word_wrap=False
        )
        
        code_panel = Panel(
            syntax,
            title="🐍 Generated Python Code",
            title_align="left",
            border_style=MOSAICX_COLORS['accent'],
            padding=(1, 2),
            width=96,
            expand=False
        )
        
        main_table.add_row("💻 Code Preview", code_panel)
        
        # Center and display the main table
        console.print(Align.center(main_table))
            
    except Exception as e:
        styled_message(f"Schema generation failed: {str(e)}", "error")
        if debug:
            console.print_exception()
        raise click.ClickException(str(e))


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
    scan: bool
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
    
    # Get schemas
    schemas = list_schemas(class_name_filter=class_name, description_filter=description)
    
    if not schemas:
        styled_message("No schemas found. Generate some schemas first!", "warning")
        return
    
    # Show beautiful card-style display of schemas for better readability
    console.print()
    console.print()
    
    from rich.panel import Panel
    from rich.columns import Columns
    
    if not schemas:
        styled_message("No schemas found. Generate some schemas first!", "warning")
        return
    
    # Create individual cards for each schema
    schema_cards = []
    
    for schema in schemas:
        # Status indicator
        status = "✅ Exists" if schema['file_exists'] else "❌ Missing"
        status_color = MOSAICX_COLORS['success'] if schema['file_exists'] else MOSAICX_COLORS['error']
        
        # Format creation date nicely
        created_at = schema['created_at']
        if 'T' in created_at:
            date_part = created_at.split('T')[0]
            time_part = created_at.split('T')[1][:8]
            formatted_date = f"{date_part} {time_part}"
        else:
            formatted_date = created_at[:19]
        
        # Create content for each schema card
        card_content = f"""[bold {MOSAICX_COLORS['primary']}]Schema ID:[/bold {MOSAICX_COLORS['primary']}] {schema['id']}
[bold {MOSAICX_COLORS['accent']}]Class:[/bold {MOSAICX_COLORS['accent']}] {schema['class_name']}
[bold {MOSAICX_COLORS['secondary']}]Description:[/bold {MOSAICX_COLORS['secondary']}] {schema['description']}

[bold {MOSAICX_COLORS['info']}]File:[/bold {MOSAICX_COLORS['info']}] {schema['file_name']}
[bold {MOSAICX_COLORS['muted']}]Model:[/bold {MOSAICX_COLORS['muted']}] {schema['model_used']}
[bold {MOSAICX_COLORS['muted']}]Created:[/bold {MOSAICX_COLORS['muted']}] {formatted_date}
[bold]Status:[/bold] [{status_color}]{status}[/{status_color}]"""
        
        # Create individual panels for each schema
        schema_panel = Panel(
            card_content,
            title=f"[bold {MOSAICX_COLORS['primary']}]{schema['class_name']}[/bold {MOSAICX_COLORS['primary']}]",
            title_align="left",
            border_style=MOSAICX_COLORS['accent'],
            padding=(1, 2),
            width=60
        )
        schema_cards.append(schema_panel)
    
    # Display title
    console.print(f"\n[bold {MOSAICX_COLORS['primary']}]📚 Generated Schemas Registry[/bold {MOSAICX_COLORS['primary']}]", justify="center")
    console.print(f"[{MOSAICX_COLORS['muted']}]Found {len(schemas)} schema(s)[/{MOSAICX_COLORS['muted']}]", justify="center")
    console.print()
    
    # Display schemas in columns (2 per row)
    for i in range(0, len(schema_cards), 2):
        row_cards = schema_cards[i:i+2]
        if len(row_cards) == 2:
            console.print(Columns(row_cards, equal=True, expand=True))
        else:
            console.print(Columns([row_cards[0], ""], equal=True, expand=True))
        console.print()  # Add spacing between rows
    
    # Show usage hint
    console.print()
    styled_message(
        f"💡 Tip: Use schema ID, filename, or file path in extract commands",
        "info"
    )


@cli.command()
@click.option("--pdf", required=True, type=click.Path(exists=True), help="Path to PDF file to extract from")
@click.option("--schema", required=True, help="Schema identifier (ID, filename, or file path)")
@click.option("--model", default=DEFAULT_LLM_MODEL, help="Ollama model name for extraction")
@click.option("--save", type=click.Path(), help="Save extracted JSON result to this path")
@click.option("--debug", is_flag=True, help="Verbose debug logs")
@click.pass_context
def extract(
    ctx: click.Context,
    pdf: str,
    schema: str,
    model: str,
    save: Optional[str],
    debug: bool
) -> None:
    """Extract structured data from PDF using a generated Pydantic schema.
    
    SCHEMA FORMATS ACCEPTED:
    • Schema ID: medicationlist_82042d1e_20250919_153054
    • Filename: medicationlist_patient_medication_list_20250919_153054.py
    • File path: mosaicx/schema/pyd/patientvitals_20250919_151730.py
    
    MODEL COMPATIBILITY:
    ⭐ gpt-oss:120b (default) | 🟢 mistral:latest | 🟢 qwen2.5-coder:32b | 🟢 deepseek-r1:8b | 🔴 gpt-oss:20b (not working)
    """
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        styled_message(f"Extracting from: {pdf}", "info")
        styled_message(f"Using schema: {schema}", "info")
        styled_message(f"Using model: {model}", "info")
    
    try:
        # Resolve schema reference to actual file path
        resolved_schema_path = _resolve_schema_reference(schema)
        if not resolved_schema_path:
            raise click.ClickException(f"Could not find schema: {schema}")
        
        if verbose:
            styled_message(f"Resolved schema to: {resolved_schema_path}", "info")
        
        # Get the class name from the schema registry
        all_schemas = list_schemas()
        schema_class_name = None
        
        # Convert resolved path to string for comparison
        resolved_path_str = str(resolved_schema_path)
        
        # Find the schema in registry by file path
        for schema_info in all_schemas:
            if resolved_path_str == schema_info['file_path']:
                schema_class_name = schema_info['class_name']
                break
        
        if not schema_class_name:
            raise click.ClickException(f"Could not find class name for schema: {schema}")
        
        if verbose:
            styled_message(f"Using schema class: {schema_class_name}", "info")
        
        # Perform extraction using the resolved file path instead of class name
        result = extract_from_pdf(pdf, str(resolved_schema_path), model, save)
        
        # Display results beautifully
        console.print()
        styled_message(f"📋 Extraction results based on schema: {schema}", "primary", center=True)
        console.print()
        
        # Create a beautiful table to display the extracted data
        from rich.table import Table
        data_table = Table(
            show_lines=False,
            border_style=MOSAICX_COLORS["secondary"],
            header_style=f"bold {MOSAICX_COLORS['primary']}"
        )
        
        data_table.add_column("Field", style=MOSAICX_COLORS["info"], no_wrap=True)
        data_table.add_column("Extracted Value", style=MOSAICX_COLORS["accent"])
        
        # Add rows for each field in the result
        result_dict = result.model_dump()
        for field_name, value in result_dict.items():
            # Format value for display
            if value is None:
                display_value = "[dim]Not found[/dim]"
            elif isinstance(value, list) or isinstance(value, dict):
                display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            else:
                display_value = str(value)
            
            data_table.add_row(field_name, display_value)
        
        console.print(Align.center(data_table))
        
        # Show file save info if saved
        if save:
            console.print()
            console.print()
            styled_message("📁 EXTRACTION SAVED", "accent", center=True)
            console.print()
            styled_message(f"JSON: {Path(save).name}", "primary", center=True)
        
        if verbose and debug:
            console.print()
            styled_message("Raw extracted data:", "secondary", center=True)
            console.print()
            from rich.json import JSON
            console.print(JSON(result.model_dump_json(indent=2)))
            
    except ExtractionError as e:
        styled_message(f"Extraction failed: {str(e)}", "error")
        if debug:
            console.print_exception()
        raise click.ClickException(str(e))
    except Exception as e:
        styled_message(f"Unexpected error: {str(e)}", "error")
        if debug:
            console.print_exception()
        raise click.ClickException(str(e))


def main(args: Optional[List[str]] = None) -> None:
    """Main entry point for the MOSAICX CLI application."""
    cli(args)


if __name__ == "__main__":
    main()
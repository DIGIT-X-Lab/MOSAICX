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
natural language processing, and data validation using the schema_builder module
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
Uses schema_builder.py as the core working prototype for all schema operations.

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
    • schema_builder: Core schema generation engine (working prototype)
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
from rich.table import Table
from .schema_builder import (
    induce_schemaspec_with_ollama,
    compile_schema_to_model,
    generate_model_py,
    fields_table
)
from .extractor import extract_from_pdf, ExtractionError

# Import metadata from constants
from .constants import (
    APPLICATION_NAME,
    APPLICATION_VERSION as __version__,
    AUTHOR_NAME as __author__,
    AUTHOR_EMAIL as __email__,
    DEFAULT_LLM_MODEL,
    MOSAICX_COLORS,
    PACKAGE_SCHEMA_JSON_DIR,
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
@click.option("--desc", required=True, help="Natural language description of fields you want")
@click.option("--model", default=DEFAULT_LLM_MODEL, help="Ollama model name")
@click.option("--example", multiple=True, help="Optional example text/report (can pass multiple)")
@click.option("--save-schema", type=click.Path(), help="Write induced SchemaSpec JSON to this path")
@click.option("--save-model", type=click.Path(), help="Write generated Pydantic class .py to this path")
@click.option("--debug", is_flag=True, help="Verbose debug logs")
@click.pass_context
def generate(
    ctx: click.Context, 
    desc: str, 
    model: str, 
    example: tuple, 
    save_schema: Optional[str], 
    save_model: Optional[str], 
    debug: bool
) -> None:
    """Generate Pydantic schemas from natural language descriptions."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        styled_message(f"Generating schema using model: {model}", "info")
        styled_message(f"Description: {desc}", "info")
    
    try:
        # Use schema_builder as the core engine
        with console.status(f"[{MOSAICX_COLORS['primary']}]Inducing schema...", spinner="dots"):
            spec = induce_schemaspec_with_ollama(
                model, 
                desc, 
                list(example) if example else None, 
                debug=debug
            )
        
        # Auto-generate default filenames based on schema name
        schema_name = spec.name.lower().replace(" ", "_")
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        default_json_name = f"{schema_name}_{timestamp}.json"
        default_py_name = f"{schema_name}_{timestamp}.py"
        
        # Determine save paths (use defaults if not specified)
        json_save_path = save_schema if save_schema else Path(PACKAGE_SCHEMA_JSON_DIR) / default_json_name
        py_save_path = save_model if save_model else Path(PACKAGE_SCHEMA_PYD_DIR) / default_py_name
        
        # Ensure directories exist
        Path(json_save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(py_save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Compile to runtime model
        with console.status(f"[{MOSAICX_COLORS['accent']}]Compiling Pydantic model...", spinner="dots"):
            Model = compile_schema_to_model(spec)
        
        # Display the model name prominently without a banner
        console.print()
        styled_message(f"✨ Schema Model: {Model.__name__} ✨", "primary", center=True)
        console.print()
        
        # Display the fields table with proper spacing
        table = fields_table(spec)
        console.print(Align.center(table))
        
        # Generate and save files
        class_code = generate_model_py(spec)
        
        # Save schema JSON
        import json
        Path(json_save_path).write_text(json.dumps(spec.model_dump(), indent=2))
        
        # Save Python code
        Path(py_save_path).write_text(class_code)
        
        # Show file save results prominently at the end
        console.print()
        console.print()
        styled_message("📁 FILES SAVED", "accent", center=True)
        console.print()
        
        # Create aligned file output
        from rich.table import Table
        file_table = Table.grid(padding=1)
        file_table.add_column(style=f"bold {MOSAICX_COLORS['secondary']}", justify="right")
        file_table.add_column(style=MOSAICX_COLORS['primary'])
        
        file_table.add_row("JSON", Path(json_save_path).name)
        file_table.add_row("Pydantic", Path(py_save_path).name)
        
        console.print(Align.center(file_table))
        
        if verbose:
            console.print()
            styled_message("Generated Python Code:", "secondary", center=True)
            console.print()
            from rich.syntax import Syntax
            syntax = Syntax(class_code, "python", theme="dracula", line_numbers=True, 
                          background_color=MOSAICX_COLORS["muted"])
            console.print(Align.center(syntax))
            
    except Exception as e:
        styled_message(f"Schema generation failed: {str(e)}", "error")
        if debug:
            console.print_exception()
        raise click.ClickException(str(e))


@cli.command()
@click.option("--pdf", required=True, type=click.Path(exists=True), help="Path to PDF file to extract from")
@click.option("--schema", required=True, help="Name of the Pydantic schema model to use")
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
    """Extract structured data from PDF using a generated Pydantic schema."""
    verbose = ctx.obj.get('verbose', False)
    
    if verbose:
        styled_message(f"Extracting from: {pdf}", "info")
        styled_message(f"Using schema: {schema}", "info")
        styled_message(f"Using model: {model}", "info")
    
    try:
        # Perform extraction
        result = extract_from_pdf(pdf, schema, model, save)
        
        # Display results beautifully
        console.print()
        console.print()
        styled_message(f"📋 Extraction Results: {schema}", "primary", center=True)
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
            elif isinstance(value, (list, dict)):
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
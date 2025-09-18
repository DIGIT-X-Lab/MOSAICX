"""
MOSAICX Main Module - Application Entry Point and Core Functionality

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Overview:
---------
This is the main module for the MOSAICX application, serving as the primary entry
point for all medical data processing workflows. It orchestrates the integration
of various components including schema validation, data extraction, intelligent
structuring, and analysis pipelines for medical imaging and clinical data.

Core Functionality:
------------------
• Application initialization and configuration management
• Command-line interface with argument parsing and validation
• Workflow orchestration for medical data processing pipelines
• Schema discovery and Pydantic model registration
• Integration with display, logging, and error handling systems
• Plugin architecture for extensible functionality

Architecture:
------------
The main module follows a modular architecture pattern, coordinating between
specialized subsystems while maintaining clear separation of concerns. It provides
both programmatic API access and command-line interface functionality for
different usage scenarios.

Usage Examples:
--------------
Command-line usage:
    $ mosaicx --help
    $ mosaicx extract --description "Patient data with demographics" --output patient_schema
    $ mosaicx extract --description "DICOM metadata fields" --output dicom --domain imaging
    $ mosaicx list-schemas

Programmatic usage:
    >>> from mosaicx import MosaicX
    >>> app = MosaicX()
    >>> app.process_medical_data(input_file="data.json")
    >>> available_schemas = app.list_available_schemas()

Dependencies:
------------
External Libraries:
    • rich-click (^1.0.0): Enhanced command-line interface framework
    • pydantic (^2.0.0): Data validation and settings management
    • rich (^13.0.0): Terminal output formatting and progress tracking
    • ollama: LLM integration for natural language processing

Internal Modules:
    • mosaicx.display: Terminal interface and banner display
    • mosaicx.nl_to_schema: Natural language to schema conversion
    • mosaicx.schema: Schema management and Pydantic model registry
    • mosaicx.utils: Utility functions and helper methods
    • mosaicx.config: Configuration management and settings

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

import sys
from pathlib import Path
from typing import List, Optional

import rich_click as click
from rich.console import Console
from rich.panel import Panel

# MOSAICX imports
from display import show_main_banner, console, styled_message
from nl_to_schema import generate_schema_from_nl

# Module metadata
__version__ = "1.0.0"
__author__ = "Lalith Kumar Shiyam Sundar, PhD"
__email__ = "Lalith.shiyam@med.uni-muenchen.de"

# Configure rich-click for better CLI appearance
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="MOSAICX")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    **MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction**
    
    A comprehensive toolkit for medical data processing, validation, and analysis.
    
    Use `mosaicx COMMAND --help` for detailed command information.
    """
    if ctx.invoked_subcommand is None:
        show_main_banner()
        styled_message("Use --help to see available commands", "info")
    
    if verbose:
        styled_message("Verbose mode enabled", "info")


@cli.command()
@click.option(
    "--description", "-d", 
    required=True,
    help="Natural language description of the schema to generate"
)
@click.option(
    "--output", "-o",
    required=True, 
    help="Output filename for the generated schema (without extension)"
)
@click.option(
    "--domain",
    default="medical",
    type=click.Choice(["medical", "imaging", "analysis", "core"]),
    help="Domain category for schema organization"
)
@click.option(
    "--example", "-e",
    multiple=True,
    help="Example data to inform schema generation (can specify multiple)"
)
@click.option(
    "--model", "-m",
    default="gpt-oss:120b", 
    help="Ollama model to use for schema generation"
)
def extract(
    description: str,
    output: str, 
    domain: str,
    example: List[str],
    model: str
) -> None:
    """
    **Extract schema from natural language description**
    
    Generate Pydantic models and JSON schemas from natural language descriptions
    using AI-powered analysis. The generated files will be automatically organized
    in the MOSAICX schema directory structure.
    
    **Examples:**
    
    ```bash
    # Basic usage
    mosaicx extract -d "Patient demographics with age, gender, ID" -o patient_info
    
    # With domain specification
    mosaicx extract -d "DICOM metadata fields" -o dicom --domain imaging
    
    # With examples
    mosaicx extract -d "Lab results" -o lab_data -e "glucose: 95 mg/dL" -e "hemoglobin: 14.2 g/dL"
    
    # Custom model
    mosaicx extract -d "Radiology report structure" -o radiology --model qwen2:7b-instruct
    ```
    """
    try:
        console.print(
            Panel.fit(
                f"[bold cyan]Generating schema: {output}[/bold cyan]\n"
                f"Domain: {domain}\n"
                f"Model: {model}\n"
                f"Description: {description}",
                title="MOSAICX Schema Extraction",
                style="cyan"
            )
        )
        
        # Convert examples tuple to list
        example_list = list(example) if example else None
        
        # Generate schema
        saved_paths = generate_schema_from_nl(
            description=description,
            domain=domain,
            schema_name=output,
            examples=example_list,
            model=model,
            workspace_root=Path.cwd()
        )
        
        # Display success message with file paths
        success_panel = Panel.fit(
            f"[bold green]✓ Schema generated successfully![/bold green]\n\n"
            f"📄 JSON Schema: {saved_paths['json']}\n"
            f"🐍 Pydantic Model: {saved_paths['pydantic']}\n\n"
            f"Files are organized in: mosaicx/schema/{domain}/",
            title="Generation Complete",
            style="green"
        )
        console.print(success_panel)
        
    except Exception as e:
        styled_message(f"Schema extraction failed: {e}", "error")
        sys.exit(1)


@cli.command()
def list_schemas() -> None:
    """
    **List all available schemas in the MOSAICX registry**
    
    Display all Pydantic models and JSON schemas currently available
    in the MOSAICX schema directory structure.
    """
    styled_message("Schema listing functionality will be implemented with schema registry", "info")


def main() -> None:
    """Main entry point for the MOSAICX CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        styled_message("\nOperation cancelled by user", "warning")
        sys.exit(1)
    except Exception as e:
        styled_message(f"Unexpected error: {e}", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()
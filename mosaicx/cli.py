# mosaicx/cli.py
"""
MOSAICX CLI v2 -- Click command skeleton.

Provides the ``mosaicx`` console entry-point declared in pyproject.toml as
``mosaicx.cli:cli``.  Every command/group below is a stub that prints its
name and arguments; real logic will be wired in later tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

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
    console.print(f"[bold]extract[/bold] document={document} template={template} optimized={optimized}")


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
    console.print(
        f"[bold]batch[/bold] input_dir={input_dir} output_dir={output_dir} "
        f"template={template} formats={formats} workers={workers} "
        f"completeness_threshold={completeness_threshold} resume={resume}"
    )


# ---------------------------------------------------------------------------
# template (group)
# ---------------------------------------------------------------------------


@cli.group()
def template() -> None:
    """Manage extraction templates."""


@template.command("create")
def template_create() -> None:
    """Create a new extraction template."""
    console.print("[bold]template create[/bold]")


@template.command("list")
def template_list() -> None:
    """List available templates."""
    console.print("[bold]template list[/bold]")


@template.command("validate")
def template_validate() -> None:
    """Validate a template file."""
    console.print("[bold]template validate[/bold]")


# ---------------------------------------------------------------------------
# schema (group)
# ---------------------------------------------------------------------------


@cli.group()
def schema() -> None:
    """Manage Pydantic schemas."""


@schema.command("generate")
def schema_generate() -> None:
    """Generate a Pydantic schema from a description."""
    console.print("[bold]schema generate[/bold]")


@schema.command("list")
def schema_list() -> None:
    """List registered schemas."""
    console.print("[bold]schema list[/bold]")


@schema.command("refine")
def schema_refine() -> None:
    """Refine an existing schema."""
    console.print("[bold]schema refine[/bold]")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dir", "directory", type=click.Path(exists=False, path_type=Path), help="Directory of reports.")
@click.option("--patient", type=str, default=None, help="Patient identifier.")
@click.option("--format", "formats", type=str, multiple=True, help="Output format(s).")
def summarize(
    directory: Optional[Path],
    patient: Optional[str],
    formats: tuple[str, ...],
) -> None:
    """Summarize clinical reports for a patient."""
    console.print(f"[bold]summarize[/bold] dir={directory} patient={patient} formats={formats}")


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
@click.option("--workers", type=int, default=4, show_default=True, help="Number of parallel workers.")
def deidentify(
    document: Optional[Path],
    directory: Optional[Path],
    mode: str,
    workers: int,
) -> None:
    """De-identify clinical documents."""
    console.print(f"[bold]deidentify[/bold] document={document} dir={directory} mode={mode} workers={workers}")


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--pipeline", type=str, default=None, help="Pipeline to optimize.")
@click.option("--trainset", type=click.Path(exists=False, path_type=Path), help="Training dataset path.")
@click.option("--valset", type=click.Path(exists=False, path_type=Path), help="Validation dataset path.")
@click.option("--budget", type=int, default=50, show_default=True, help="Optimization budget (number of trials).")
@click.option("--save", type=click.Path(path_type=Path), help="Save optimized program to this path.")
def optimize(
    pipeline: Optional[str],
    trainset: Optional[Path],
    valset: Optional[Path],
    budget: int,
    save: Optional[Path],
) -> None:
    """Optimize a DSPy pipeline."""
    console.print(
        f"[bold]optimize[/bold] pipeline={pipeline} trainset={trainset} "
        f"valset={valset} budget={budget} save={save}"
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
    console.print("[bold]MOSAICX Configuration[/bold]")
    for field_name, field_value in cfg.model_dump().items():
        console.print(f"  {field_name}: {field_value}")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value."""
    console.print(f"[bold]config set[/bold] {key}={value}")

# mosaicx/cli_theme.py
"""DigiTx-inspired terminal theme for MOSAICX CLI.

Coral & greige palette drawn from thedigitxlab.com:
  - cfonts block banner with coral-to-greige gradient
  - Numbered section headers ("01 · SECTION NAME")
  - Rounded panels with warm greige borders
  - Status badges with reverse styling
  - Works in both light and dark terminal modes
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ── Brand ─────────────────────────────────────────────────────────

BRAND = "M O S A I C X"
TAGLINE = "Medical Computational Suite for Advanced Intelligent Extraction"
ORG = "DIGITX Lab \u00b7 LMU Radiology \u00b7 LMU Munich"

# ── Palette ───────────────────────────────────────────────────────
# Coral (#E87461) and greige (#B5A89A) — warm, distinctive, visible
# on both light and dark terminal backgrounds.

CORAL = "#E87461"
GREIGE = "#B5A89A"
MUTED = "dim"


# ── Banner ────────────────────────────────────────────────────────


def print_banner(version: str, console: Console, lm: str = "", lm_cheap: str = "") -> None:
    """Print the MOSAICX banner using cfonts with coral-to-greige gradient."""
    try:
        from cfonts import render

        output = render(
            "MOSAICX",
            font="block",
            gradient=[CORAL, GREIGE],
            transition=True,
            space=False,
        )
        # Leading blank line + indent each line by 1 extra space to align with subtitle
        indented = "\n".join(" " + line for line in output.split("\n"))
        console.file.write("\n" + indented)
    except ImportError:
        # Fallback without cfonts
        console.print(f"\n  [bold {CORAL}]{BRAND}[/bold {CORAL}]\n")

    # Subtitle below the banner
    console.print()
    console.print(f"  [{GREIGE}]{TAGLINE}[/{GREIGE}]")
    console.print(f"  [{MUTED}]v{version} \u00b7 {ORG}[/{MUTED}]")

    # Status strip: model badges
    if lm or lm_cheap:
        lm_short = lm.split("/", 1)[-1] if "/" in lm else lm
        lm_cheap_short = lm_cheap.split("/", 1)[-1] if "/" in lm_cheap else lm_cheap
        rule = "\u2500" * len(TAGLINE)
        console.print(f"  [{GREIGE}]{rule}[/{GREIGE}]")
        console.print(
            f"  [reverse {CORAL}] chonk [/reverse {CORAL}] [{MUTED}]\u25b8[/{MUTED}] [{CORAL}]{lm_short}[/{CORAL}]"
            f"   "
            f"[reverse {GREIGE}] smol [/reverse {GREIGE}] [{MUTED}]\u25b8[/{MUTED}] [{CORAL}]{lm_cheap_short}[/{CORAL}]"
        )

    console.print()


def print_version(version: str, console: Console) -> None:
    """Print a compact branded version line."""
    t = Text()
    t.append(BRAND, style=f"bold {CORAL}")
    t.append(f"  v{version}", style=MUTED)
    console.print(t)


# ── Section headers ──────────────────────────────────────────────


def section(
    title: str,
    console: Console,
    number: str | None = None,
    uppercase: bool = True,
) -> None:
    """Print a DigiTx-style numbered section header."""
    console.print()
    t = Text()
    if number:
        t.append(f"  {number}", style=f"bold {CORAL}")
        t.append(" \u00b7 ", style=MUTED)
    else:
        t.append("  ", style="")
    display = title.upper() if uppercase else title
    t.append(display, style="bold")
    console.print(t)
    rule = "\u2500" * len(TAGLINE)
    console.print(f"  {rule}", style=GREIGE)


# ── Tables ───────────────────────────────────────────────────────


def make_table(title: str | None = None, **kwargs: object) -> Table:
    """Create a table with DigiTx styling (rounded, greige border)."""
    return Table(
        title=title,
        box=box.ROUNDED,
        border_style=GREIGE,
        title_style=f"bold {CORAL}",
        header_style="bold",
        padding=(0, 1),
        **kwargs,
    )


def make_kv_table() -> Table:
    """Create a headerless two-column key\u2013value table."""
    t = make_table(show_header=False)
    t.add_column("Key", style=f"bold {CORAL}", no_wrap=True)
    t.add_column("Value")
    return t


def make_clean_table(**kwargs: object) -> Table:
    """Create a borderless table with dim headers and clean spacing."""
    return Table(
        box=None,
        show_edge=False,
        pad_edge=False,
        header_style=MUTED,
        padding=(0, 2),
        **kwargs,
    )


# ── Inline badges & tags ────────────────────────────────────────


def badge(label: str, variant: str = "default") -> str:
    """Return Rich markup for a filled status badge."""
    colors = {
        "default": CORAL,
        "stable": "green",
        "dev": "yellow",
        "warn": "yellow",
        "error": "red",
    }
    c = colors.get(variant, CORAL)
    return f"[reverse {c}] {label} [/reverse {c}]"


def tag(label: str) -> str:
    """Return Rich markup for a dim tag/chip."""
    return f"[{GREIGE}]\u00b7[/{GREIGE}] {label}"


# ── Status lines ─────────────────────────────────────────────────


def info(msg: str) -> str:
    """Info-level status line (coral arrow, dim text)."""
    return f"  [{CORAL}]\u203a[/{CORAL}] [{MUTED}]{msg}[/{MUTED}]"


def ok(msg: str) -> str:
    """Success status line (green check)."""
    return f"  [bold green]\u2713[/bold green] {msg}"


def warn(msg: str) -> str:
    """Warning status line (yellow bang)."""
    return f"  [bold yellow]![/bold yellow] [yellow]{msg}[/yellow]"


def err(msg: str) -> str:
    """Error status line (red cross)."""
    return f"  [bold red]\u2717[/bold red] {msg}"


# ── Progress helpers ────────────────────────────────────────────


@contextmanager
def spinner(label: str, console: Console) -> Generator[None, None, None]:
    """Coral dots spinner for indeterminate operations (LLM calls, OCR, etc.)."""
    p = Progress(
        TextColumn(" "),
        SpinnerColumn("dots", style=Style(color=CORAL)),
        TextColumn(f"[{MUTED}]{label}[/{MUTED}]"),
        console=console,
        transient=True,
    )
    with p:
        p.add_task(label, total=None)
        yield


@contextmanager
def progress(total: int, label: str, console: Console) -> Generator[object, None, None]:
    """Coral progress bar for countable operations (batch docs, pages, iterations)."""
    p = Progress(
        TextColumn(f"  [{CORAL}]\u25b8[/{CORAL}]"),
        BarColumn(complete_style=Style(color=CORAL), finished_style=Style(color=CORAL)),
        MofNCompleteColumn(),
        TextColumn(f"[{MUTED}]{label}[/{MUTED}]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with p:
        task = p.add_task(label, total=total)
        yield lambda: p.advance(task)

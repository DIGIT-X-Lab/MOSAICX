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


def print_banner(
    version: str,
    console: Console,
    lm: str = "",
    *,
    num_ctx: int = 0,
    inference_engine: str = "",
    ocr_engine: str = "",
    ocr_langs: list[str] | None = None,
) -> None:
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

    # Status strip
    if lm or ocr_engine:
        rule = "\u2500" * len(TAGLINE)
        console.print(f"  [{GREIGE}]{rule}[/{GREIGE}]")

    # Two-column status strip using Rich Table for proper alignment
    if lm or ocr_engine:
        from rich.padding import Padding

        lm_short = (lm.split("/", 1)[-1] if "/" in lm else lm) if lm else ""

        # Build each cell as a Text object
        # Row 1: pill + name
        model_r1 = Text()
        ocr_r1 = Text()
        if lm:
            model_r1.append("\ue0b6", style=f"{CORAL}")
            model_r1.append(" Model ", style=f"reverse {CORAL}")
            model_r1.append("\ue0b4", style=f"{CORAL}")
            model_r1.append(f"  {lm_short}", style="bold")
        if ocr_engine:
            ocr_r1.append("\ue0b6", style=f"{GREIGE}")
            ocr_r1.append("  OCR  ", style=f"reverse {GREIGE}")
            ocr_r1.append("\ue0b4", style=f"{GREIGE}")
            ocr_r1.append(f"  {ocr_engine}", style="bold")

        # Row 2: details indented to align under the name (after the pill)
        model_r2 = Text()
        ocr_r2 = Text()
        if lm:
            # Spacer matching pill visual width so details align under model name
            model_r2.append("\ue0b6 Model \ue0b4  ", style="")  # invisible pill-width spacer
            model_r2.stylize("", 0, len(model_r2))  # clear any inherited style
            # Actually use a simpler approach: just prepend spaces matching the pill
            model_r2 = Text()
            model_r2.append("           ", style="")  # visual width of pill + gap
            has_detail = False
            if num_ctx > 0:
                ctx_human = f"{num_ctx // 1000}k"
                model_r2.append("ctx ", style=MUTED)
                model_r2.append(ctx_human, style=GREIGE)
                has_detail = True
            if inference_engine:
                if has_detail:
                    model_r2.append("  \u00b7  ", style=GREIGE)
                model_r2.append("via ", style=MUTED)
                model_r2.append(inference_engine, style=GREIGE)
        if ocr_engine and ocr_langs:
            ocr_r2.append("           ", style="")  # same spacer
            ocr_r2.append("langs ", style=MUTED)
            ocr_r2.append(", ".join(ocr_langs), style=GREIGE)

        # Borderless 2-column table — Rich handles alignment
        t = Table(box=None, show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
        t.add_column("model", no_wrap=True)
        t.add_column("ocr", no_wrap=True)
        t.add_row(model_r1, ocr_r1)
        if model_r2.plain.strip() or ocr_r2.plain.strip():
            t.add_row(model_r2, ocr_r2)
        console.print(Padding(t, (0, 0, 0, 2)))

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
    t = Table(
        box=box.ROUNDED,
        border_style=GREIGE,
        show_header=False,
        show_edge=False,
        padding=(0, 1, 0, 2),
    )
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
def spinner(
    label: str,
    console: Console,
    quips: list[str] | None = None,
    quip_interval: float = 3.0,
) -> Generator[None, None, None]:
    """Coral dots spinner for indeterminate operations (LLM calls, OCR, etc.).

    If *quips* is provided, the label text cycles through them every
    *quip_interval* seconds while the spinner runs.
    """
    p = Progress(
        TextColumn(" "),
        SpinnerColumn("dots", style=Style(color=CORAL)),
        TextColumn(f"[{MUTED}]{{task.description}}[/{MUTED}]"),
        console=console,
        transient=True,
    )
    with p:
        task_id = p.add_task(label, total=None)
        if quips:
            import random
            import threading

            stop = threading.Event()

            def _rotate():
                while not stop.wait(quip_interval):
                    # Extract base prefix (everything before "...")
                    base = label.rsplit("...", 1)[0]
                    quip = random.choice(quips)
                    p.update(task_id, description=f"{base}... {quip}")

            t = threading.Thread(target=_rotate, daemon=True)
            t.start()
            try:
                yield
            finally:
                stop.set()
                t.join(timeout=1)
        else:
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


# ── Wave spinner ───────────────────────────────────────────────


class LiveSpinner:
    """Animated spinner with traveling coral-to-greige wave gradient.

    Writes directly to the TTY with ANSI true-color escape codes so each
    character gets its own RGB color as the wave sweeps across the text.
    Optionally rotates through *quips* to keep the user entertained.
    """

    _FRAMES = ["\u22c5", "+", "\u2733", "+", "\u22c5"]  # ⋅ + ✳ + ⋅

    def __init__(self, console: Console) -> None:
        self.console = console
        self.text = ""
        self.running = False
        self.thread = None
        self._coral = (232, 116, 97)
        self._greige = (181, 168, 154)
        self._reset = "\033[0m"
        self._frame_idx = 0
        self._wave_pos = 0

    def start(self, text: str = "", show_elapsed: bool = False) -> None:
        import threading
        import time as _time

        self.text = text
        self.running = True
        self._frame_idx = 0
        self._wave_pos = 0
        self._show_elapsed = show_elapsed
        self._start_time = _time.monotonic()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def update(self, text: str) -> None:
        self.text = text

    def _lerp_color(self, t: float) -> str:
        t = max(0.0, min(1.0, t))
        r = int(self._coral[0] + (self._greige[0] - self._coral[0]) * t)
        g = int(self._coral[1] + (self._greige[1] - self._coral[1]) * t)
        b = int(self._coral[2] + (self._greige[2] - self._coral[2]) * t)
        return f"\033[38;2;{r};{g};{b}m"

    def _animate(self) -> None:
        import math
        import os
        import sys
        import time

        try:
            tty = open("/dev/tty", "w")  # noqa: SIM115
        except OSError:
            tty = sys.stderr

        wave_width = 32

        while self.running:
            frame = self._FRAMES[self._frame_idx % len(self._FRAMES)]
            text = self.text

            try:
                cols = os.get_terminal_size().columns
            except OSError:
                cols = 80
            max_text = cols - 4
            if len(text) > max_text:
                text = text[: max_text - 1] + "\u2026"

            text_len = len(text) if text else 1

            colored = ""
            for i, char in enumerate(text):
                dist = abs(
                    i - (self._wave_pos % (text_len + wave_width * 2)) + wave_width
                )
                if dist < wave_width:
                    t = (1 - math.cos(math.pi * dist / wave_width)) / 2
                else:
                    t = 1.0
                colored += self._lerp_color(t) + char

            spinner_color = self._lerp_color(0)
            elapsed_str = ""
            if self._show_elapsed:
                elapsed = time.monotonic() - self._start_time
                mins, secs = divmod(int(elapsed), 60)
                greige = self._lerp_color(1.0)
                elapsed_str = f" {greige}[{mins}:{secs:02d}]{self._reset} "
            line = f"\r\033[K  {spinner_color}{frame}{self._reset}{elapsed_str}{colored}{self._reset}"
            tty.write(line)
            tty.flush()

            self._frame_idx += 1
            self._wave_pos += 3
            time.sleep(0.18)

        if tty not in (sys.stderr, sys.stdout):
            tty.close()

    def stop(self) -> None:
        import sys

        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        try:
            tty = open("/dev/tty", "w")  # noqa: SIM115
            tty.write("\r\033[K")
            tty.flush()
            tty.close()
        except OSError:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()


@contextmanager
def wave_spinner(
    label: str,
    console: Console,
    quips: list[str] | None = None,
    quip_interval: float = 3.0,
    show_elapsed: bool = True,
) -> Generator[None, None, None]:
    """Wave-gradient spinner with optional rotating quips and elapsed time.

    Uses the LiveSpinner (coral-to-greige traveling wave) instead of
    Rich's built-in spinner. If *quips* is provided, the displayed text
    cycles through them (no prefix, just the quip).
    """
    s = LiveSpinner(console)
    initial = quips[0] if quips else label
    s.start(initial, show_elapsed=show_elapsed)
    if quips:
        import random
        import threading

        stop = threading.Event()

        def _rotate() -> None:
            while not stop.wait(quip_interval):
                s.update(random.choice(quips))

        t = threading.Thread(target=_rotate, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join(timeout=1)
            s.stop()
    else:
        try:
            yield
        finally:
            s.stop()

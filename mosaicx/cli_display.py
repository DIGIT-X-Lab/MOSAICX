# mosaicx/cli_display.py
"""Smart display helpers for extraction output.

Renders extraction dicts as clean borderless tables instead of raw JSON.
Dispatches on data shape: flat dicts → KV tables, lists of dicts → column
tables, nested dicts → subsections.  Works for all extraction paths
(radiology mode, auto mode, template mode).
"""

from __future__ import annotations

import json
import math
import re
from enum import Enum
from typing import Any

from rich.console import Console
from rich.padding import Padding

from . import cli_theme as theme

# Columns to show first when rendering a list-of-dicts table.
_PRIORITY_COLS = [
    "anatomy", "observation", "description", "statement",
    "measurement", "category", "actionable", "severity",
    "change_from_prior",
]

# Columns that are mostly internal IDs — skip in display.
_SKIP_COLS = {"radlex_id", "template_field_id", "finding_refs"}

_MAX_COLUMNS = 5
_MAX_ROWS = 20
_TRUNCATED_ROWS = 15


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_extracted_data(data: dict[str, Any], console: Console) -> None:
    """Render extraction output with smart formatting.

    Unwraps the ``{"extracted": ...}`` wrapper that schema/auto/template
    modes produce, so the user sees fields directly.
    """
    if not data:
        console.print(theme.info("No data extracted"))
        return

    keys = set(data.keys())

    # Unwrap template mode: {"extracted": {flat dict}}
    if keys == {"extracted"} and isinstance(data["extracted"], dict):
        _render_dict_items(data["extracted"], console, depth=0)
        return

    # Unwrap auto mode: {"extracted": {...}, "inferred_schema": {...}}
    if "extracted" in keys and "inferred_schema" in keys:
        if isinstance(data["extracted"], dict):
            _render_dict_items(data["extracted"], console, depth=0)
        if isinstance(data["inferred_schema"], dict):
            _render_subsection("Inferred Template", data["inferred_schema"], console, depth=0)
        return

    # Mode output (radiology, pathology) — render top-level keys directly
    _render_dict_items(data, console, depth=0)


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _render_dict_items(data: dict[str, Any], console: Console, depth: int) -> None:
    """Render a dict by grouping adjacent scalars into a KV table and
    dispatching non-scalar values to specialised renderers."""
    scalar_buf: list[tuple[str, Any]] = []

    def flush_scalars() -> None:
        if not scalar_buf:
            return
        t = theme.make_clean_table(show_header=False)
        t.add_column("Key", style=f"bold {theme.CORAL}", no_wrap=True)
        t.add_column("Value")
        for k, v in scalar_buf:
            t.add_row(_pretty_key(k), _format_value(v))
        console.print(Padding(t, (0, 0, 0, 2)))
        scalar_buf.clear()

    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            scalar_buf.append((key, value))
        elif isinstance(value, dict):
            flush_scalars()
            _render_subsection(key, value, console, depth)
        elif isinstance(value, list):
            flush_scalars()
            if not value:
                continue
            if all(isinstance(item, dict) for item in value):
                _render_list_table(key, value, console)
            else:
                _render_bullet_list(key, value, console)
        else:
            scalar_buf.append((key, value))

    flush_scalars()


def _render_subsection(key: str, data: dict[str, Any], console: Console, depth: int) -> None:
    """Render a nested dict as a subsection with its own header."""
    if depth > 2:
        # Too deep — fall back to compact JSON
        theme.section(_pretty_key(key), console, uppercase=False)
        console.print(Padding(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            (0, 0, 0, 2),
        ))
        return

    is_flat = all(
        v is None or isinstance(v, (str, int, float, bool))
        for v in data.values()
    )

    if is_flat:
        theme.section(_pretty_key(key), console, uppercase=False)
        t = theme.make_clean_table(show_header=False)
        t.add_column("Key", style=f"bold {theme.CORAL}", no_wrap=True)
        t.add_column("Value")
        for k, v in data.items():
            if v is None:
                continue
            t.add_row(_pretty_key(k), _format_value(v))
        console.print(Padding(t, (0, 0, 0, 2)))
    else:
        theme.section(_pretty_key(key), console, uppercase=False)
        _render_dict_items(data, console, depth + 1)


def _render_list_table(key: str, items: list[dict], console: Console) -> None:
    """Render a list of dicts as a table with auto-discovered columns."""
    # Discover all keys
    all_keys: list[str] = []
    seen: set[str] = set()
    for item in items:
        for k in item:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Count non-null values per column
    col_fill: dict[str, int] = {k: 0 for k in all_keys}
    for item in items:
        for k in all_keys:
            if item.get(k) is not None:
                col_fill[k] += 1

    # Filter: skip columns that are null in most rows for generic tables.
    # Keep sparse clinical level-finding columns visible to avoid hiding key findings.
    sparse_findings = (
        len(items) <= 12
        and any(
            marker in {k.lower() for k in all_keys}
            for marker in {"level", "disc_bulge_type", "disc_protrusion", "disc_location"}
        )
    )
    threshold = 1 if sparse_findings else max(1, math.ceil(len(items) * 0.2))
    viable = [k for k in all_keys if col_fill[k] >= threshold and k not in _SKIP_COLS]

    # Sort: priority columns first, then remainder
    priority = [k for k in _PRIORITY_COLS if k in viable]
    rest = [k for k in viable if k not in priority]
    cols = (priority + rest)[:_MAX_COLUMNS]

    if not cols:
        return

    theme.section(f"{_pretty_key(key)} ({len(items)})", console, uppercase=False)

    t = theme.make_clean_table()
    for col in cols:
        style = f"bold {theme.CORAL}" if col == cols[0] else None
        t.add_column(_pretty_key(col), style=style)

    display_items = items[:_TRUNCATED_ROWS] if len(items) > _MAX_ROWS else items
    for item in display_items:
        row = [_format_value(item.get(col)) for col in cols]
        t.add_row(*row)

    console.print(Padding(t, (0, 0, 0, 2)))

    if len(items) > _MAX_ROWS:
        console.print(theme.info(f"Showing {_TRUNCATED_ROWS} of {len(items)} rows"))
    if len(viable) > _MAX_COLUMNS:
        console.print(theme.info(f"Showing {_MAX_COLUMNS} of {len(viable)} columns — use -o for full data"))


def _render_bullet_list(key: str, items: list, console: Console) -> None:
    """Render a list of scalars as a numbered list."""
    theme.section(f"{_pretty_key(key)} ({len(items)})", console, uppercase=False)
    for i, item in enumerate(items[:_MAX_ROWS], 1):
        console.print(f"  [{theme.CORAL}]{i}.[/{theme.CORAL}] {_format_value(item)}")
    if len(items) > _MAX_ROWS:
        console.print(theme.info(f"... and {len(items) - _MAX_ROWS} more"))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pretty_key(key: str) -> str:
    """Convert snake_case key to Title Case."""
    return key.replace("_", " ").title()


def _humanize_enum_like_value(text: str) -> str:
    """Convert internal enum/member tokens into user-facing labels."""
    value = " ".join(str(text or "").split())
    if not value:
        return value

    looks_internal = "." in value and any(ch.isupper() for ch in value.split(".", 1)[0])
    if looks_internal:
        value = value.rsplit(".", 1)[-1]

    # Preserve common vertebral level ranges (e.g. C3-C4, C7-T1).
    level_match = re.fullmatch(r"\s*([A-Za-z]?\d+)\s*[-–]\s*([A-Za-z]?\d+)\s*", value)
    if level_match:
        return f"{level_match.group(1).upper()}-{level_match.group(2).upper()}"

    # Convert snake/camel style labels into readable words.
    value = value.replace("_", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = " ".join(value.split())
    if not value:
        return value
    if value.isupper() and len(value) <= 4:
        return value
    return value[0].upper() + value[1:]


def _format_value(val: Any) -> str:
    """Format a value for display."""
    if val is None:
        return "\u2014"
    if isinstance(val, bool):
        return f"[green]Yes[/green]" if val else f"[dim]No[/dim]"
    if isinstance(val, Enum):
        enum_value = getattr(val, "value", val)
        return _humanize_enum_like_value(str(enum_value))
    if isinstance(val, dict):
        # Measurement-like: {"value": 12.0, "unit": "mm", ...}
        if "value" in val and "unit" in val:
            s = f"{val['value']} {val['unit']}"
            if val.get("dimension"):
                s += f" ({val['dimension']})"
            return s
        # Generic small dict — inline
        return json.dumps(val, default=str)
    if isinstance(val, list):
        return ", ".join(_format_value(v) for v in val)
    if isinstance(val, str):
        return _humanize_enum_like_value(val)
    return str(val)


# ---------------------------------------------------------------------------
# Performance metrics display
# ---------------------------------------------------------------------------


def render_completeness(completeness: dict[str, Any], console: Console) -> None:
    """Render a completeness panel with score and missing fields.

    Parameters
    ----------
    completeness:
        Dict with keys from :class:`ReportCompleteness` (as produced by
        ``dataclasses.asdict()``): ``overall``, ``required_coverage``,
        ``optional_coverage``, ``field_coverage``, ``filled_fields``,
        ``total_fields``, ``missing_required``.
    console:
        Rich console instance.
    """
    from rich.panel import Panel
    from rich import box

    overall = completeness.get("overall", 0.0)
    pct = overall * 100

    # Color based on score
    if pct >= 80:
        color = "green"
    elif pct >= 60:
        color = "yellow"
    else:
        color = "red"

    filled = completeness.get("filled_fields", 0)
    total = completeness.get("total_fields", 0)
    req_cov = completeness.get("required_coverage", 0.0)
    missing = completeness.get("missing_required", [])

    # Count required vs optional from total
    req_total = sum(
        1 for f in completeness.get("fields", []) if f.get("required")
    )
    req_filled = sum(
        1 for f in completeness.get("fields", [])
        if f.get("required") and f.get("filled")
    )

    lines: list[str] = []
    lines.append(
        f"  [{color}]{pct:.0f}%[/{color}]"
        f" [{theme.GREIGE}]overall completeness[/{theme.GREIGE}]"
        f"  [{theme.MUTED}]({filled}/{total} fields filled)[/{theme.MUTED}]"
    )
    if req_total:
        lines.append(
            f"  [{theme.CORAL}]{req_filled}/{req_total}[/{theme.CORAL}]"
            f" [{theme.GREIGE}]required fields filled[/{theme.GREIGE}]"
        )
    if missing:
        lines.append("")
        lines.append(f"  [{theme.MUTED}]Missing required:[/{theme.MUTED}]")
        for name in missing:
            lines.append(f"    [{color}]-[/{color}] {name}")

    content = "\n".join(lines)

    theme.section("Completeness", console)
    console.print(
        Panel(
            content,
            box=box.ROUNDED,
            border_style=theme.GREIGE,
            padding=(0, 1),
        )
    )


def render_metrics(metrics: "PipelineMetrics", console: Console) -> None:
    """Render a compact single-line performance summary."""
    from .metrics import PipelineMetrics  # noqa: F811 — runtime guard

    if not isinstance(metrics, PipelineMetrics) or not metrics.steps:
        return

    console.print()
    console.print(
        f"  [{theme.GREIGE}]{'─' * len(theme.TAGLINE)}[/{theme.GREIGE}]"
    )
    n = len(metrics.steps)
    console.print(
        f"  [{theme.CORAL}]{metrics.total_tokens:,}[/{theme.CORAL}]"
        f" [{theme.GREIGE}]tokens[/{theme.GREIGE}]"
        f" [{theme.GREIGE}]·[/{theme.GREIGE}]"
        f" [{theme.CORAL}]{metrics.total_duration_s:.1f}s[/{theme.CORAL}]"
        f" [{theme.GREIGE}]·[/{theme.GREIGE}]"
        f" [{theme.CORAL}]{n}[/{theme.CORAL}]"
        f" [{theme.GREIGE}]step{'s' if n != 1 else ''}[/{theme.GREIGE}]"
    )

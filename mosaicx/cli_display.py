# mosaicx/cli_display.py
"""Smart display helpers for extraction output.

Renders extraction dicts as clean borderless tables instead of raw JSON.
Dispatches on data shape: flat dicts → KV tables, lists of dicts → column
tables, nested dicts → subsections.  Works for all extraction paths
(radiology mode, schema mode, auto mode, template mode).
"""

from __future__ import annotations

import json
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

    # Unwrap schema / template mode: {"extracted": {flat dict}}
    if keys == {"extracted"} and isinstance(data["extracted"], dict):
        _render_dict_items(data["extracted"], console, depth=0)
        return

    # Unwrap auto mode: {"extracted": {...}, "inferred_schema": {...}}
    if "extracted" in keys and "inferred_schema" in keys:
        if isinstance(data["extracted"], dict):
            _render_dict_items(data["extracted"], console, depth=0)
        if isinstance(data["inferred_schema"], dict):
            _render_subsection("Inferred Schema", data["inferred_schema"], console, depth=0)
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

    # Filter: skip columns that are null in >80% of rows, and skip internal IDs
    threshold = len(items) * 0.2
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


def _format_value(val: Any) -> str:
    """Format a value for display."""
    if val is None:
        return "\u2014"
    if isinstance(val, bool):
        return f"[green]Yes[/green]" if val else f"[dim]No[/dim]"
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
        return ", ".join(str(v) for v in val)
    return str(val)

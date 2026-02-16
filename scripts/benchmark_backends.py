#!/usr/bin/env python3
"""Benchmark LLM backends for MOSAICX extraction.

Runs the same extraction task across multiple OpenAI-compatible backends
and produces a comparison table of tokens, timing, and throughput.

Usage:
    python scripts/benchmark_backends.py \
        --document /tmp/test_report.txt \
        --mode radiology \
        --runs 3

    # Add a custom backend
    python scripts/benchmark_backends.py \
        --document /tmp/test_report.txt \
        --backend "custom=openai/my-model@http://localhost:9000/v1"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Backend definitions
# ---------------------------------------------------------------------------

BACKENDS: list[dict[str, str]] = [
    {
        "name": "vllm-mlx",
        "model": "openai/default",
        "api_base": "http://localhost:8000/v1",
    },
    {
        "name": "ollama",
        "model": "openai/gpt-oss:20b",
        "api_base": "http://localhost:11434/v1",
    },
    {
        "name": "llama-cpp",
        "model": "openai/default",
        "api_base": "http://localhost:8080/v1",
    },
    {
        "name": "sglang",
        "model": "openai/default",
        "api_base": "http://localhost:30000/v1",
    },
]


@dataclass
class RunResult:
    """Metrics from a single extraction run."""

    duration_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class BackendResult:
    """Aggregated results for one backend across all runs."""

    name: str
    model: str
    api_base: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def avg_duration(self) -> float:
        return statistics.mean(r.duration_s for r in self.runs)

    @property
    def min_duration(self) -> float:
        return min(r.duration_s for r in self.runs)

    @property
    def avg_tokens(self) -> float:
        return statistics.mean(r.total_tokens for r in self.runs)

    @property
    def avg_tok_per_sec(self) -> float:
        rates = [r.total_tokens / r.duration_s for r in self.runs if r.duration_s > 0]
        return statistics.mean(rates) if rates else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probe_backend(api_base: str, timeout: float = 3.0) -> bool:
    """Return True if the backend responds to GET /models."""
    url = f"{api_base}/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _parse_backend_arg(arg: str) -> dict[str, str]:
    """Parse ``name=model@api_base`` into a backend dict.

    Examples:
        "my-server=openai/my-model@http://localhost:9000/v1"
        "custom=openai/default@http://10.0.0.5:8000/v1"
    """
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"Invalid backend format: {arg!r}. Expected name=model@api_base"
        )
    name, rest = arg.split("=", 1)
    if "@" not in rest:
        raise argparse.ArgumentTypeError(
            f"Invalid backend format: {arg!r}. Expected name=model@api_base"
        )
    model, api_base = rest.rsplit("@", 1)
    return {"name": name.strip(), "model": model.strip(), "api_base": api_base.strip()}


def _configure_dspy_for_backend(model: str, api_base: str) -> None:
    """Configure DSPy to use a specific backend."""
    import dspy

    from mosaicx.metrics import TokenTracker, make_harmony_lm, set_tracker

    lm = make_harmony_lm(model, api_key="not-needed", api_base=api_base)
    dspy.configure(lm=lm)

    tracker = TokenTracker()
    set_tracker(tracker)
    dspy.settings.usage_tracker = tracker
    dspy.settings.track_usage = True


def _run_extraction(document_text: str, mode: str) -> RunResult:
    """Run a single extraction and return metrics."""
    from mosaicx.pipelines.extraction import extract_with_mode

    _output, metrics = extract_with_mode(document_text, mode)

    if metrics is not None:
        return RunResult(
            duration_s=metrics.total_duration_s,
            input_tokens=metrics.total_input_tokens,
            output_tokens=metrics.total_output_tokens,
            total_tokens=metrics.total_tokens,
        )
    # Fallback if no metrics returned
    return RunResult(duration_s=0.0, input_tokens=0, output_tokens=0, total_tokens=0)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_results(results: list[BackendResult], mode: str, num_runs: int) -> None:
    """Print a Rich comparison table."""
    from rich.console import Console

    from mosaicx.cli_theme import CORAL, GREIGE, MUTED, make_table, ok, section, warn

    console = Console()

    section(f"Benchmark Results ({mode} · {num_runs} run{'s' if num_runs != 1 else ''})", console)

    table = make_table()
    table.add_column("Backend", no_wrap=True)
    table.add_column("Model")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Avg Time", justify="right")
    table.add_column("Fastest", justify="right")
    table.add_column("Tok/s", justify="right")

    # Sort by average duration (fastest first)
    ranked = sorted(results, key=lambda r: r.avg_duration)
    fastest_name = ranked[0].name if ranked else ""

    for res in ranked:
        model_short = res.model.split("/", 1)[-1] if "/" in res.model else res.model
        is_fastest = res.name == fastest_name
        name_style = f"bold {CORAL}" if is_fastest else ""
        marker = " [bold green]★[/bold green]" if is_fastest else ""

        table.add_row(
            f"[{name_style}]{res.name}[/{name_style}]{marker}" if name_style else res.name,
            f"[{MUTED}]{model_short}[/{MUTED}]",
            f"{res.avg_tokens:,.0f}",
            f"{res.avg_duration:.1f}s",
            f"{res.min_duration:.1f}s",
            f"{res.avg_tok_per_sec:.1f}",
        )

    console.print()
    console.print(table)
    console.print()


def _save_results(results: list[BackendResult], mode: str, num_runs: int, path: Path) -> None:
    """Save benchmark results to a JSON file."""
    data = {
        "mode": mode,
        "runs_per_backend": num_runs,
        "backends": [
            {
                "name": r.name,
                "model": r.model,
                "api_base": r.api_base,
                "avg_duration_s": round(r.avg_duration, 2),
                "min_duration_s": round(r.min_duration, 2),
                "avg_tokens": round(r.avg_tokens),
                "avg_tok_per_sec": round(r.avg_tok_per_sec, 1),
                "runs": [
                    {
                        "duration_s": round(run.duration_s, 2),
                        "input_tokens": run.input_tokens,
                        "output_tokens": run.output_tokens,
                        "total_tokens": run.total_tokens,
                    }
                    for run in r.runs
                ],
            }
            for r in sorted(results, key=lambda r: r.avg_duration)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LLM backends for MOSAICX extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/benchmark_backends.py --document report.txt\n"
            "  python scripts/benchmark_backends.py --document report.txt --runs 3 --mode pathology\n"
            '  python scripts/benchmark_backends.py --document report.txt --backend "fast=openai/m@http://host:8000/v1"\n'
        ),
    )
    parser.add_argument("--document", required=True, type=Path, help="Path to test document")
    parser.add_argument("--mode", default="radiology", help="Extraction mode (default: radiology)")
    parser.add_argument("--runs", type=int, default=1, help="Runs per backend (default: 1)")
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        dest="custom_backends",
        metavar="NAME=MODEL@URL",
        help="Custom backend: name=model@api_base (repeatable)",
    )
    parser.add_argument("--output", type=Path, help="Save results JSON to file")
    parser.add_argument("--only", help="Only run these backends (comma-separated names)")
    args = parser.parse_args()

    # Late imports so --help is fast
    from rich.console import Console

    from mosaicx.cli_theme import err, info, ok, section, spinner, warn
    from mosaicx.documents.loader import load_document

    console = Console()

    # ── Load document ────────────────────────────────────────────
    if not args.document.exists():
        console.print(err(f"Document not found: {args.document}"))
        sys.exit(1)

    section("Load Document", console, number="01")
    with spinner("Loading document…", console):
        doc = load_document(args.document)
    console.print(ok(f"{doc.source_path.name} ({len(doc.text):,} chars, {doc.page_count} page{'s' if doc.page_count != 1 else ''})"))

    # ── Build backend list ───────────────────────────────────────
    backends = list(BACKENDS)
    for raw in args.custom_backends:
        backends.append(_parse_backend_arg(raw))

    # Deduplicate by name (custom overrides built-in)
    seen: dict[str, dict[str, str]] = {}
    for b in backends:
        seen[b["name"]] = b
    backends = list(seen.values())

    # Filter --only
    if args.only:
        only_names = {n.strip() for n in args.only.split(",")}
        backends = [b for b in backends if b["name"] in only_names]
        if not backends:
            console.print(err(f"No backends match --only={args.only}"))
            sys.exit(1)

    # ── Probe backends ───────────────────────────────────────────
    section("Probe Backends", console, number="02")
    live_backends: list[dict[str, str]] = []
    for b in backends:
        reachable = _probe_backend(b["api_base"])
        if reachable:
            console.print(ok(f"{b['name']} @ {b['api_base']}"))
            live_backends.append(b)
        else:
            console.print(warn(f"{b['name']} @ {b['api_base']} — offline, skipping"))

    if not live_backends:
        console.print(err("No backends are reachable. Start at least one backend and retry."))
        sys.exit(1)

    # ── Run benchmarks ───────────────────────────────────────────
    section("Benchmark", console, number="03")
    console.print(info(f"Mode: {args.mode} · Runs per backend: {args.runs}"))
    console.print()

    all_results: list[BackendResult] = []

    for b in live_backends:
        result = BackendResult(name=b["name"], model=b["model"], api_base=b["api_base"])
        _configure_dspy_for_backend(b["model"], b["api_base"])

        for run_idx in range(args.runs):
            label = f"{b['name']} · run {run_idx + 1}/{args.runs}"

            # Fresh tracker per run so metrics don't bleed across iterations
            import dspy

            from mosaicx.metrics import TokenTracker, set_tracker

            tracker = TokenTracker()
            set_tracker(tracker)
            dspy.settings.usage_tracker = tracker

            with spinner(label, console):
                try:
                    run = _run_extraction(doc.text, args.mode)
                    result.runs.append(run)
                    console.print(ok(f"{label} — {run.duration_s:.1f}s, {run.total_tokens:,} tokens"))
                except Exception as exc:
                    console.print(err(f"{label} — failed: {exc}"))

        if result.runs:
            all_results.append(result)
        else:
            console.print(warn(f"{b['name']} — all runs failed, excluding from results"))

    if not all_results:
        console.print(err("All benchmark runs failed."))
        sys.exit(1)

    # ── Results ──────────────────────────────────────────────────
    _print_results(all_results, args.mode, args.runs)

    if args.output:
        _save_results(all_results, args.mode, args.runs, args.output)
        console.print(ok(f"Results saved to {args.output}"))


if __name__ == "__main__":
    main()

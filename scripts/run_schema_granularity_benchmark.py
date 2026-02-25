#!/usr/bin/env python3
"""Run schema granularity benchmark (baseline vs hybrid semantic gate).

This script evaluates template/schema generation quality across multiple
document styles and persists a reproducible artifact bundle under docs/runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CaseSpec:
    case_id: str
    name: str
    description: str
    text: str
    expects_repeated_structure: bool
    expects_enums: bool


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        probe = value.strip().lower()
        return probe in {"", "none", "null", "na", "n/a"}
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _configure_dspy() -> None:
    from mosaicx.config import get_config
    from mosaicx.metrics import TokenTracker, make_harmony_lm, set_tracker
    from mosaicx.runtime_env import configure_dspy_lm

    cfg = get_config()
    lm = make_harmony_lm(
        cfg.lm,
        api_key=cfg.api_key,
        api_base=cfg.api_base,
        temperature=cfg.lm_temperature,
    )
    dspy, adapter_name = configure_dspy_lm(
        lm,
        preferred_cache_dir=cfg.home_dir / ".dspy_cache",
    )
    # Disable DSPy response caching for true baseline-vs-hybrid comparisons.
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    tracker = TokenTracker()
    set_tracker(tracker)
    dspy.settings.usage_tracker = tracker
    dspy.settings.track_usage = True
    os.environ["MOSAICX_DSPY_ADAPTER_ACTIVE"] = adapter_name


def _clear_dspy_caches() -> None:
    targets = [
        Path.home() / ".dspy_cache",
        Path.home() / ".mosaicx" / ".dspy_cache",
        ROOT / ".mosaicx_runtime" / "dspy_cache",
        Path("/tmp") / "mosaicx" / "dspy_cache",
    ]
    for target in targets:
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)


def _load_cases(path: Path) -> list[CaseSpec]:
    from mosaicx.documents import load_document

    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[CaseSpec] = []
    for item in payload:
        source_path = str(item.get("source_path", "")).strip()
        text = str(item.get("text", "")).strip()
        if source_path:
            source = (ROOT / source_path).resolve()
            loaded = load_document(source)
            text = loaded.text
        if not text.strip():
            raise ValueError(f"Case {item.get('id')} has no usable text.")
        cases.append(
            CaseSpec(
                case_id=str(item["id"]),
                name=str(item["name"]),
                description=str(item["description"]),
                text=text,
                expects_repeated_structure=bool(item.get("expects_repeated_structure", False)),
                expects_enums=bool(item.get("expects_enums", False)),
            )
        )
    return cases


def _schema_stats(spec: Any) -> dict[str, Any]:
    from mosaicx.pipelines.schema_gen import _normalize_type_string  # noqa: PLC2701

    fields = []

    def walk(items: list[Any]) -> None:
        for f in items:
            fields.append(f)
            if getattr(f, "fields", None):
                walk(list(f.fields))

    walk(list(spec.fields))
    total = max(len(fields), 1)
    list_object_fields = 0
    object_fields = 0
    enum_fields = 0
    non_str_fields = 0
    for f in fields:
        t = _normalize_type_string(
            f.type,
            has_fields=bool(getattr(f, "fields", None)),
            has_enum_values=bool(getattr(f, "enum_values", None)),
        )
        if t == "list[object]":
            list_object_fields += 1
        if t in {"object", "list[object]"}:
            object_fields += 1
        if t == "enum":
            enum_fields += 1
        if t != "str":
            non_str_fields += 1
    return {
        "field_count_total": len(fields),
        "list_object_fields": list_object_fields,
        "object_fields": object_fields,
        "enum_fields": enum_fields,
        "typed_ratio": round(non_str_fields / total, 4),
    }


def _required_coverage(spec: Any, extracted: dict[str, Any]) -> float:
    required = [f.name for f in spec.fields if getattr(f, "required", True)]
    if not required:
        return 1.0
    present = sum(0 if _is_blank(extracted.get(name)) else 1 for name in required)
    return present / max(len(required), 1)


def _run_mode_case(
    *,
    mode: str,
    case: CaseSpec,
    out_dir: Path,
    hybrid_semantic_min_score: float,
    generation_context: str,
) -> dict[str, Any]:
    from mosaicx.pipelines.extraction import DocumentExtractor
    from mosaicx.pipelines.schema_gen import (
        SchemaGenerator,
        assess_schema_semantic_granularity,
        compile_schema,
    )
    from mosaicx.schemas.template_compiler import schema_spec_to_template_yaml

    generator = SchemaGenerator()
    started = time.perf_counter()
    kwargs = dict(
        description=case.description,
        example_text="",
        document_text=case.text if generation_context == "from_document" else "",
        runtime_dryrun=True,
        max_repairs=2,
    )
    if mode == "baseline":
        kwargs["semantic_min_score"] = 0.0
        kwargs["enable_semantic_gate"] = False
        kwargs["use_llm_semantic_assessor"] = False
    elif mode == "hybrid":
        kwargs["semantic_min_score"] = max(0.0, min(1.0, float(hybrid_semantic_min_score)))
        kwargs["enable_semantic_gate"] = True
        kwargs["use_llm_semantic_assessor"] = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pred = generator(**kwargs)
    elapsed_s = time.perf_counter() - started
    spec = pred.schema_spec
    stats = _schema_stats(spec)
    semantic_score, semantic_issues = assess_schema_semantic_granularity(
        spec,
        document_text=case.text,
    )

    yaml_path = out_dir / f"{case.case_id}_{mode}.yaml"
    yaml_text = schema_spec_to_template_yaml(spec)
    yaml_path.write_text(yaml_text, encoding="utf-8")

    compiled = compile_schema(spec)
    extractor = DocumentExtractor(output_schema=compiled)
    extract_started = time.perf_counter()
    extract_result = extractor(document_text=case.text)
    extract_elapsed_s = time.perf_counter() - extract_started
    extracted = getattr(extract_result, "extracted", extract_result)
    if hasattr(extracted, "model_dump"):
        extracted_payload = extracted.model_dump()
    elif isinstance(extracted, dict):
        extracted_payload = extracted
    else:
        extracted_payload = {}

    required_cov = _required_coverage(spec, extracted_payload)
    extraction_ok = required_cov >= 0.5
    repeated_ok = (not case.expects_repeated_structure) or (stats["list_object_fields"] > 0)
    enum_ok = (not case.expects_enums) or (stats["enum_fields"] > 0)

    return {
        "mode": mode,
        "generation_context": generation_context,
        "case_id": case.case_id,
        "case_name": case.name,
        "elapsed_s": round(elapsed_s, 2),
        "extract_elapsed_s": round(extract_elapsed_s, 2),
        "semantic_score": round(float(semantic_score), 4),
        "semantic_issues": semantic_issues,
        "schema_issues": list(getattr(pred, "schema_issues", []) or []),
        "runtime_dryrun_used": bool(getattr(pred, "runtime_dryrun_used", False)),
        "semantic_gate_triggered": bool(getattr(pred, "semantic_gate_applied", False)),
        "stats": stats,
        "required_coverage": round(required_cov, 4),
        "extraction_ok": extraction_ok,
        "repeated_structure_ok": repeated_ok,
        "enum_ok": enum_ok,
        "template_yaml": str(yaml_path),
    }


def _aggregate(rows: list[dict[str, Any]], mode: str, generation_context: str) -> dict[str, Any]:
    mode_rows = [
        r
        for r in rows
        if r["mode"] == mode and r.get("generation_context") == generation_context
    ]
    if not mode_rows:
        return {}
    return {
        "mode": mode,
        "generation_context": generation_context,
        "case_count": len(mode_rows),
        "semantic_score_mean": round(statistics.mean(r["semantic_score"] for r in mode_rows), 4),
        "required_coverage_mean": round(statistics.mean(r["required_coverage"] for r in mode_rows), 4),
        "extraction_success_rate": round(
            sum(1 for r in mode_rows if r["extraction_ok"]) / len(mode_rows),
            4,
        ),
        "repeated_structure_pass_rate": round(
            sum(1 for r in mode_rows if r["repeated_structure_ok"]) / len(mode_rows),
            4,
        ),
        "enum_pass_rate": round(
            sum(1 for r in mode_rows if r["enum_ok"]) / len(mode_rows),
            4,
        ),
        "mean_list_object_fields": round(
            statistics.mean(r["stats"]["list_object_fields"] for r in mode_rows),
            4,
        ),
        "mean_enum_fields": round(
            statistics.mean(r["stats"]["enum_fields"] for r in mode_rows),
            4,
        ),
        "semantic_gate_trigger_rate": round(
            sum(1 for r in mode_rows if r.get("semantic_gate_triggered")) / len(mode_rows),
            4,
        ),
    }


def _render_markdown(
    *,
    out_md: Path,
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    hybrid: dict[str, Any],
    cases_path: Path,
    generation_context: str,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = []
    lines.append("# Schema Granularity Benchmark")
    lines.append("")
    lines.append(f"- Generated: {ts}")
    lines.append(f"- Cases: `{cases_path}`")
    lines.append("- Modes: `baseline` (semantic gate off) vs `hybrid` (semantic gate on + DSPy assessor)")
    lines.append(f"- Generation context: `{generation_context}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Baseline | Hybrid | Delta |")
    lines.append("|---|---:|---:|---:|")
    for metric in [
        "semantic_score_mean",
        "required_coverage_mean",
        "extraction_success_rate",
        "repeated_structure_pass_rate",
        "enum_pass_rate",
        "mean_list_object_fields",
        "mean_enum_fields",
        "semantic_gate_trigger_rate",
    ]:
        b = float(baseline.get(metric, 0.0))
        h = float(hybrid.get(metric, 0.0))
        lines.append(f"| {metric} | {b:.4f} | {h:.4f} | {h - b:+.4f} |")
    lines.append("")
    lines.append("## Per Case")
    lines.append("")
    lines.append(
        "| Case | Mode | Semantic | ReqCov | ExtractOK | RepeatOK | EnumOK | ListObj | EnumFields | Gen(s) | Extract(s) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {case_id} | {mode} | {semantic_score:.4f} | {required_coverage:.4f} | {extraction_ok} | "
            "{repeated_structure_ok} | {enum_ok} | {list_obj} | {enum_fields} | {elapsed_s:.2f} | {extract_elapsed_s:.2f} |".format(
                case_id=row["case_id"],
                mode=row["mode"],
                semantic_score=row["semantic_score"],
                required_coverage=row["required_coverage"],
                extraction_ok="yes" if row["extraction_ok"] else "no",
                repeated_structure_ok="yes" if row["repeated_structure_ok"] else "no",
                enum_ok="yes" if row["enum_ok"] else "no",
                list_obj=row["stats"]["list_object_fields"],
                enum_fields=row["stats"]["enum_fields"],
                elapsed_s=row["elapsed_s"],
                extract_elapsed_s=row["extract_elapsed_s"],
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Benchmarks use the configured local DSPy LM and adapter policy.")
    lines.append("- DSPy cache should be cleared before this run for cold-start comparability.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run schema granularity benchmark.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "tests" / "datasets" / "evaluation" / "schema_granularity_cases.json",
        help="Path to benchmark case JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: docs/runs/<timestamp>-schema-granularity-benchmark).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on case count (0 = all).",
    )
    parser.add_argument(
        "--hybrid-semantic-min-score",
        type=float,
        default=0.60,
        help="Semantic threshold for hybrid mode (0..1).",
    )
    parser.add_argument(
        "--generation-context",
        type=str,
        choices=("from_document", "describe_only"),
        default="from_document",
        help="Schema generation context. `from_document` uses source text. `describe_only` uses description only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        out_dir = ROOT / "docs" / "runs" / f"{run_id}-schema-granularity-benchmark"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from mosaicx.config import get_config
    from mosaicx.runtime_env import check_openai_endpoint_ready

    cfg = get_config()
    endpoint = check_openai_endpoint_ready(
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        ping_model=cfg.lm,
        timeout_s=20.0,
    )
    if not endpoint.ok:
        raise RuntimeError(
            f"LLM endpoint is not ready: {endpoint.reason} (api_base={endpoint.api_base})"
        )

    _configure_dspy()
    cases = _load_cases(args.cases)
    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    rows: list[dict[str, Any]] = []
    for case in cases:
        for mode in ("baseline", "hybrid"):
            _clear_dspy_caches()
            print(
                f"[schema-benchmark] case={case.case_id} mode={mode} context={args.generation_context}",
                flush=True,
            )
            row = _run_mode_case(
                mode=mode,
                case=case,
                out_dir=out_dir,
                hybrid_semantic_min_score=args.hybrid_semantic_min_score,
                generation_context=args.generation_context,
            )
            rows.append(row)

    baseline = _aggregate(rows, "baseline", args.generation_context)
    hybrid = _aggregate(rows, "hybrid", args.generation_context)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_file": str(args.cases),
        "adapter_active": os.environ.get("MOSAICX_DSPY_ADAPTER_ACTIVE"),
        "llm_endpoint": endpoint.to_dict(),
        "generation_context": args.generation_context,
        "baseline": baseline,
        "hybrid": hybrid,
        "rows": rows,
    }
    out_json = out_dir / "schema_granularity_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_md = out_dir / "schema_granularity_report.md"
    _render_markdown(
        out_md=out_md,
        rows=rows,
        baseline=baseline,
        hybrid=hybrid,
        cases_path=args.cases,
        generation_context=args.generation_context,
    )

    print(f"[schema-benchmark] results={out_json}")
    print(f"[schema-benchmark] report={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

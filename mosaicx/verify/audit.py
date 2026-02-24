"""Level 3: RLM-based audit verification (30-90 seconds, multi-step).

Uses ``dspy.RLM`` to let a language model programmatically cross-reference
every extracted value against the source document(s).  The LLM iterates
through fields, searches the source text for evidence, and builds a
detailed audit report.

DSPy is imported lazily inside ``run_audit()`` so the module can be
imported without dspy fully configured.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from .models import FieldVerdict, Issue
from .parse_utils import parse_json_like

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification tools (bound to source_text & extraction at call time)
# ---------------------------------------------------------------------------


class _OutlinesEvidence(BaseModel):
    excerpt: str | None = None
    chunk_id: int | None = None
    start: int | None = None
    end: int | None = None
    score: float | None = None
    evidence_type: str | None = None
    source: str | None = None

    model_config = {"extra": "ignore"}


class _OutlinesFieldVerdict(BaseModel):
    field_path: str
    status: str
    claimed_value: str | None = None
    source_value: str | None = None
    detail: str | None = None
    evidence: _OutlinesEvidence | None = None

    model_config = {"extra": "ignore"}


class _OutlinesAuditReport(BaseModel):
    field_verdicts: list[_OutlinesFieldVerdict]
    omissions: list[str] = []
    summary: str | None = None

    model_config = {"extra": "ignore"}


def _compact_text(value: Any, *, max_chars: int = 420) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _normalize_local_api_base(base_url: str) -> str:
    return (
        str(base_url or "")
        .replace("://localhost", "://127.0.0.1")
        .replace("://[::1]", "://127.0.0.1")
    )


def _normalize_model_name_for_openai_compatible(model_name: str) -> str:
    model_name = str(model_name or "").strip()
    if "/" in model_name:
        provider, rest = model_name.split("/", 1)
        if provider.strip().lower() in {"openai", "ollama"} and rest.strip():
            return rest.strip()
    return model_name


def _run_outlines_structured_report(
    *,
    prompt: str,
    max_tokens: int = 1400,
) -> dict[str, Any] | None:
    """Generate schema-constrained JSON using Outlines over OpenAI-compatible API."""
    try:
        import openai
        import outlines

        from mosaicx.config import get_config

        cfg = get_config()
        base_url = _normalize_local_api_base(str(cfg.api_base or "http://127.0.0.1:8000/v1"))
        model_name = _normalize_model_name_for_openai_compatible(str(cfg.lm or ""))
        if not model_name:
            model_name = "mlx-community/gpt-oss-120b-4bit"

        client = openai.OpenAI(base_url=base_url, api_key=(cfg.api_key or "ollama"))
        model = outlines.from_openai(client, model_name=model_name)
        generator = outlines.Generator(model, outlines.json_schema(_OutlinesAuditReport))
        raw = generator(
            prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        parsed = parse_json_like(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False))
        if isinstance(parsed, dict):
            return parsed
        logger.warning("Outlines structured report was non-dict (%s)", type(parsed).__name__)
    except Exception as exc:
        logger.warning("Outlines structured report generation failed: %s", exc)
    return None


def _looks_like_lm_unconfigured_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    markers = (
        "no lm is loaded",
        "please configure the lm",
        "dspy.configure(lm=",
        "no lm configured",
    )
    return any(marker in msg for marker in markers)


def _is_structured_parse_failure(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    markers = (
        "jsonadapter",
        "adapterparseerror",
        "cannot be serialized to a json object",
        "cannot be serialized to json",
        "failed to parse lm response",
        "parse",
    )
    return any(marker in msg for marker in markers)


def _should_attempt_outlines_recovery(exc: Exception) -> bool:
    """Use Outlines only when DSPy failed at structured serialization/parsing.

    Non-serialization failures (e.g. LM not configured) should bubble so engine
    fallback semantics remain stable.
    """
    if _looks_like_lm_unconfigured_error(exc):
        return False
    return _is_structured_parse_failure(exc)


def _recover_claim_audit_with_outlines(
    *,
    claim: str,
    source_text: str,
    source_manifest: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Recover claim audit report when DSPy RLM cannot serialize JSON."""
    search_chunks = _make_search_source_chunks(chunks)
    queries = [claim]
    queries.extend(re.findall(r"\b\d{2,3}\s*/\s*\d{2,3}\b", claim))
    queries.extend(re.findall(r"\d+(?:\.\d+)?", claim))

    evidence_rows: list[str] = []
    seen: set[tuple[int, str]] = set()
    for query in queries[:6]:
        for hit in search_chunks(query, top_k=3):
            if not isinstance(hit, dict):
                continue
            chunk_id = int(hit.get("chunk_id", -1))
            snippet = _compact_text(hit.get("snippet") or hit.get("exact_match"), max_chars=260)
            if chunk_id < 0 or not snippet:
                continue
            key = (chunk_id, snippet)
            if key in seen:
                continue
            seen.add(key)
            evidence_rows.append(
                f"- chunk_id={chunk_id} score={hit.get('score', 0)} snippet={snippet}"
            )
            if len(evidence_rows) >= 12:
                break
        if len(evidence_rows) >= 12:
            break

    source_excerpt = _compact_text(source_text, max_chars=12000)
    prompt = (
        "You are a medical verification auditor. "
        "Return ONLY a JSON object that matches the schema.\n\n"
        "Task: Verify one claim against source evidence.\n"
        "Rules:\n"
        "- status must be one of verified, mismatch, unsupported, not_checked.\n"
        "- Use mismatch only if source has a conflicting concrete value.\n"
        "- Use unsupported if value is not found in source evidence.\n"
        "- For blood pressure, prefer measured BP readings over reference ranges.\n"
        "- Include evidence excerpt and source_value when available.\n\n"
        f"Claim:\n{claim}\n\n"
        f"Source manifest:\n{source_manifest}\n\n"
        f"Evidence hits:\n{chr(10).join(evidence_rows) if evidence_rows else '- none'}\n\n"
        f"Source excerpt:\n{source_excerpt}\n"
    )
    return _run_outlines_structured_report(prompt=prompt, max_tokens=1200)


def _recover_extraction_audit_with_outlines(
    *,
    source_text: str,
    extraction: dict[str, Any],
    source_manifest: str,
) -> dict[str, Any] | None:
    """Recover extraction audit report when DSPy RLM cannot serialize JSON."""
    extraction_json = _compact_text(
        json.dumps(extraction, ensure_ascii=False, default=str),
        max_chars=22000,
    )
    source_excerpt = _compact_text(source_text, max_chars=22000)
    prompt = (
        "You are a medical extraction auditor. "
        "Return ONLY a JSON object that matches the schema.\n\n"
        "Task: verify extraction values against source text.\n"
        "Rules:\n"
        "- status must be one of verified, mismatch, unsupported, not_checked.\n"
        "- Compare concrete values (numbers, units, categorical values) exactly.\n"
        "- Add omissions when clinically relevant source content is missing.\n"
        "- Keep field_verdicts concise but grounded with evidence excerpts.\n\n"
        f"Source manifest:\n{source_manifest}\n\n"
        f"Extraction JSON:\n{extraction_json}\n\n"
        f"Source excerpt:\n{source_excerpt}\n"
    )
    return _run_outlines_structured_report(prompt=prompt, max_tokens=1800)


def _chunk_source_text(
    source_text: str,
    *,
    chunk_chars: int = 1800,
    overlap_chars: int = 220,
    max_chunks: int = 500,
) -> list[dict[str, Any]]:
    """Split source text into overlapping chunks for long-doc audits."""
    if chunk_chars < 200:
        chunk_chars = 200
    overlap_chars = max(0, min(overlap_chars, chunk_chars - 1))

    chunks: list[dict[str, Any]] = []
    text = source_text or ""
    if not text:
        return chunks

    start = 0
    chunk_id = 0
    n_chars = len(text)
    while start < n_chars and chunk_id < max_chunks:
        end = min(n_chars, start + chunk_chars)
        chunk_text = text[start:end]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "char_count": len(chunk_text),
                "text": chunk_text,
            }
        )
        if end >= n_chars:
            break
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end
        start = next_start
        chunk_id += 1
    return chunks


def _build_source_manifest(
    source_text: str,
    chunks: list[dict[str, Any]],
    *,
    max_chunks: int = 24,
) -> str:
    """Create compact chunk index text for RLM prompt context."""
    if not chunks:
        return "Source manifest: empty source text."

    lines = [
        f"Source manifest: total_chars={len(source_text)}; chunks={len(chunks)}",
        "Use list_source_chunks/get_source_chunk/search_source_chunks tools for complete coverage.",
    ]
    for chunk in chunks[:max(1, max_chunks)]:
        preview = " ".join(str(chunk.get("text") or "").split())
        preview = preview[:140]
        lines.append(
            f"- chunk_id={chunk['chunk_id']} range={chunk['start']}:{chunk['end']} preview={preview}"
        )
    if len(chunks) > max_chunks:
        lines.append(f"- ... {len(chunks) - max_chunks} additional chunks omitted from manifest.")
    return "\n".join(lines)


def _make_list_source_chunks(chunks: list[dict[str, Any]]):
    """Create a tool that lists available chunk ids and ranges."""

    def list_source_chunks(limit: int = 200) -> list[dict[str, Any]]:
        """List source chunks with id/range metadata for navigation."""
        out: list[dict[str, Any]] = []
        for chunk in chunks[: max(1, limit)]:
            preview = " ".join(str(chunk.get("text") or "").split())[:120]
            out.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "char_count": chunk["char_count"],
                    "preview": preview,
                }
            )
        return out

    return list_source_chunks


def _make_get_source_chunk(chunks: list[dict[str, Any]]):
    """Create a tool that fetches a full chunk by id."""

    chunk_map = {int(c["chunk_id"]): c for c in chunks}

    def get_source_chunk(chunk_id: int) -> dict[str, Any]:
        """Get full text for a given source chunk id."""
        chunk = chunk_map.get(int(chunk_id))
        if chunk is None:
            return {
                "chunk_id": int(chunk_id),
                "error": "chunk_not_found",
                "available_chunk_ids": sorted(chunk_map.keys())[:200],
            }
        return {
            "chunk_id": chunk["chunk_id"],
            "start": chunk["start"],
            "end": chunk["end"],
            "char_count": chunk["char_count"],
            "text": chunk["text"],
        }

    return get_source_chunk


def _make_search_source_chunks(chunks: list[dict[str, Any]]):
    """Create a tool that searches across chunked source text."""

    def search_source_chunks(
        query: str,
        context_chars: int = 300,
        top_k: int = 12,
    ) -> list[dict[str, Any]]:
        """Search all chunks for a query and return chunk-aware ranked matches.

        Uses both exact phrase and term-level matching so paraphrased questions
        can still surface relevant chunk evidence.
        """
        q = (query or "").strip()
        if not q:
            return []
        query_lower = q.lower()
        terms = [t for t in re.findall(r"[a-z0-9]+", query_lower) if len(t) >= 2]
        out: list[dict[str, Any]] = []
        for chunk in chunks:
            text = str(chunk.get("text") or "")
            if not text:
                continue
            text_lower = text.lower()

            exact_pos = text_lower.find(query_lower)
            term_hits = 0
            term_coverage = 0
            best_term_pos = -1
            best_term = ""
            for term in terms:
                pattern = re.compile(rf"\b{re.escape(term)}\b")
                matches = list(pattern.finditer(text_lower))
                if not matches:
                    continue
                term_hits += len(matches)
                term_coverage += 1
                if best_term_pos < 0:
                    best_term_pos = matches[0].start()
                    best_term = term

            score = (8 if exact_pos >= 0 else 0) + (term_coverage * 2) + term_hits
            if score <= 0:
                continue

            anchor_pos = exact_pos if exact_pos >= 0 else best_term_pos
            anchor_len = len(q) if exact_pos >= 0 else len(best_term)
            if anchor_pos < 0:
                anchor_pos = 0
                anchor_len = min(24, len(text))
            snip_start = max(0, anchor_pos - context_chars // 2)
            snip_end = min(len(text), anchor_pos + max(anchor_len, 1) + context_chars // 2)
            snippet = text[snip_start:snip_end]
            out.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "start": chunk["start"] + anchor_pos,
                    "end": chunk["start"] + anchor_pos + max(anchor_len, 1),
                    "snippet": snippet,
                    "exact_match": text[anchor_pos: anchor_pos + max(anchor_len, 1)],
                    "score": score,
                    "evidence_type": "text_chunk",
                }
            )
        out.sort(key=lambda row: int(row.get("score") or 0), reverse=True)
        if not out:
            return [{"chunk_id": -1, "note": f"'{q}' not found in source chunks"}]
        return out[: max(1, top_k)]

    return search_source_chunks

def _make_search_source(source_text: str):
    """Create a search_source tool bound to the given source text."""

    def search_source(query: str, context_chars: int = 300) -> list[dict[str, Any]]:
        """Search the source document for a keyword or phrase.

        Returns matching snippets with surrounding context.
        Use this to find evidence for or against extracted values.
        """
        query_lower = query.lower()
        text_lower = source_text.lower()
        results: list[dict[str, Any]] = []
        start = 0

        while True:
            idx = text_lower.find(query_lower, start)
            if idx == -1:
                break
            # Extract snippet with context
            snip_start = max(0, idx - context_chars // 2)
            snip_end = min(len(source_text), idx + len(query) + context_chars // 2)
            snippet = source_text[snip_start:snip_end]
            results.append({
                "match_position": idx,
                "snippet": snippet,
                "exact_match": source_text[idx:idx + len(query)],
            })
            start = idx + 1
            if len(results) >= 10:
                break

        if not results:
            return [{"match_position": -1, "snippet": "", "exact_match": "", "note": f"'{query}' not found in source document"}]
        return results

    return search_source


def _make_get_field(extraction: dict[str, Any]):
    """Create a get_field tool bound to the given extraction."""

    def get_field(field_path: str) -> str:
        """Get the value of a specific field from the extraction.

        Use dotted paths like 'findings[0].measurement.value'.
        Returns the JSON-encoded value, or an error message if not found.
        """
        parts = field_path.replace("[", ".[").split(".")
        current: Any = extraction
        try:
            for part in parts:
                if not part:
                    continue
                if part.startswith("[") and part.endswith("]"):
                    idx = int(part[1:-1])
                    current = current[idx]
                elif isinstance(current, dict):
                    current = current[part]
                else:
                    return f"Error: cannot access '{part}' on {type(current).__name__}"
            return json.dumps(current, default=str)
        except (KeyError, IndexError, TypeError) as exc:
            return f"Error: {exc}"

    return get_field


def _make_list_fields(extraction: dict[str, Any]):
    """Create a list_fields tool bound to the given extraction."""

    def list_fields() -> list[dict[str, str]]:
        """List all fields in the extraction with their values.

        Returns a list of {path, value, type} dicts for systematic checking.
        """
        fields: list[dict[str, str]] = []

        def _walk(obj: Any, prefix: str) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    path = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (dict, list)):
                        _walk(v, path)
                    elif v is not None:
                        fields.append({
                            "path": path,
                            "value": json.dumps(v, default=str),
                            "type": type(v).__name__,
                        })
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _walk(item, f"{prefix}[{i}]")

        _walk(extraction, "")
        return fields

    return list_fields


def _make_search_numbers(source_text: str):
    """Create a tool that finds all numbers/measurements in the source."""

    def search_numbers() -> list[dict[str, Any]]:
        """Find all numbers and measurements in the source document.

        Returns a list of {value, unit, context} dicts.
        Useful for cross-referencing extracted measurements.
        """
        # Match numbers with optional units
        pattern = r'(\d+(?:\.\d+)?)\s*(mm|cm|m|kg|g|mg|ml|mL|%|years?|months?|days?|hours?|minutes?)?'
        matches = re.finditer(pattern, source_text)
        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        for m in matches:
            value = m.group(1)
            unit = m.group(2) or ""
            key = f"{value}{unit}"
            if key in seen:
                continue
            seen.add(key)
            # Get surrounding context
            start = max(0, m.start() - 80)
            end = min(len(source_text), m.end() + 80)
            context = source_text[start:end]
            results.append({
                "value": value,
                "unit": unit,
                "context": context,
            })

        return results

    return search_numbers


# ---------------------------------------------------------------------------
# Main audit function
# ---------------------------------------------------------------------------

def run_audit(
    source_text: str,
    extraction: dict[str, Any],
) -> tuple[list[Issue], list[FieldVerdict]]:
    """Run RLM-based audit verification on an extraction.

    The RLM iterates through fields, searches the source for evidence,
    cross-references values, checks for omissions, and produces a
    detailed audit report.

    Parameters
    ----------
    source_text:
        Original document text.
    extraction:
        Structured extraction dict to audit.

    Returns
    -------
    tuple[list[Issue], list[FieldVerdict]]
        Issues found and per-field verdicts from the RLM audit.
    """
    import dspy

    chunks = _chunk_source_text(source_text)
    source_manifest = _build_source_manifest(source_text, chunks)

    # Build tools bound to this source/extraction
    search_source = _make_search_source(source_text)
    list_source_chunks = _make_list_source_chunks(chunks)
    get_source_chunk = _make_get_source_chunk(chunks)
    search_source_chunks = _make_search_source_chunks(chunks)
    get_field = _make_get_field(extraction)
    list_fields = _make_list_fields(extraction)
    search_numbers = _make_search_numbers(source_text)

    tools = [
        dspy.Tool(
            search_source,
            name="search_source",
            desc="Search the source document for a keyword/phrase. Returns matching snippets with context.",
        ),
        dspy.Tool(
            list_source_chunks,
            name="list_source_chunks",
            desc="List chunk ids and ranges for the source document.",
        ),
        dspy.Tool(
            get_source_chunk,
            name="get_source_chunk",
            desc="Fetch full text for a given chunk id.",
        ),
        dspy.Tool(
            search_source_chunks,
            name="search_source_chunks",
            desc="Search across chunked source text and return chunk-aware matches.",
        ),
        dspy.Tool(
            get_field,
            name="get_field",
            desc="Get a field value from the extraction by path (e.g. 'findings[0].measurement.value').",
        ),
        dspy.Tool(
            list_fields,
            name="list_fields",
            desc="List all fields in the extraction with paths, values, and types.",
        ),
        dspy.Tool(
            search_numbers,
            name="search_numbers",
            desc="Find all numbers/measurements in the source document with context.",
        ),
    ]

    # Build extraction summary for the RLM prompt
    extraction_json = json.dumps(extraction, indent=2, default=str)

    class AuditExtraction(dspy.Signature):
        """You are a medical document verification auditor.
        Systematically verify that every value in the extraction is supported
        by the source document.

        Steps:
        1. Use list_source_chunks() to understand source coverage.
        2. Use list_fields() to get all extracted fields.
        3. For each field, use search_source_chunks()/get_source_chunk() and search_source() for evidence.
        4. Use search_numbers() to cross-reference all numeric values.
        5. Check for omissions: important content in the source not in extraction.
        6. Return a JSON audit report.

        audit_report MUST be a JSON object:
        {"field_verdicts": [{"field_path": "...", "status": "verified|mismatch|unsupported",
        "source_value": "value from source or null",
        "detail": "brief explanation",
        "evidence": {"excerpt": "...", "chunk_id": 12, "start": 420, "end": 460, "score": 7.0}}],
        "omissions": ["important source content not in extraction"],
        "summary": "one sentence overall assessment"}
        """

        source_manifest: str = dspy.InputField(desc="Chunk manifest for source navigation")
        extraction_json: str = dspy.InputField(desc="JSON extraction to verify")
        audit_report: str = dspy.OutputField(desc="JSON audit report")

    rlm = dspy.RLM(
        AuditExtraction,
        max_iterations=25,
        tools=tools,
    )

    try:
        prediction = rlm(source_manifest=source_manifest, extraction_json=extraction_json)
        return _parse_audit_report(prediction.audit_report)
    except Exception as exc:
        if not _should_attempt_outlines_recovery(exc):
            raise
        logger.warning("DSPy RLM extraction audit failed; attempting Outlines recovery: %s", exc)
        recovered = _recover_extraction_audit_with_outlines(
            source_text=source_text,
            extraction=extraction,
            source_manifest=source_manifest,
        )
        if recovered is None:
            raise
        issues, verdicts = _parse_audit_report(json.dumps(recovered, ensure_ascii=False))
        issues.append(
            Issue(
                type="audit_structured_recovery",
                field="verify.audit",
                detail=(
                    "Recovered audit output with Outlines constrained JSON generation "
                    f"after DSPy RLM serialization failure: {exc}"
                ),
                severity="warning",
            )
        )
        return issues, verdicts


def _parse_audit_report(
    raw_report: str,
) -> tuple[list[Issue], list[FieldVerdict]]:
    """Parse the RLM audit report into Issues and FieldVerdicts."""
    issues: list[Issue] = []
    field_verdicts: list[FieldVerdict] = []

    def _to_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            if isinstance(value.get("exact_match"), str) and value["exact_match"].strip():
                return value["exact_match"]
            if isinstance(value.get("snippet"), str) and value["snippet"].strip():
                return value["snippet"]
        try:
            return json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            return str(value)

    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    report = parse_json_like(raw_report)
    if report is None:
        logger.warning("RLM audit returned non-JSON: %s", raw_report[:300])
        return issues, field_verdicts

    # Accept a bare list of verdict dicts by wrapping into report format.
    if isinstance(report, list):
        report = {"field_verdicts": report, "omissions": []}

    if not isinstance(report, dict):
        logger.warning("RLM audit returned non-dict: %s", type(report).__name__)
        return issues, field_verdicts

    # Parse field verdicts
    for v in report.get("field_verdicts", []):
        if not isinstance(v, dict):
            continue

        field_path = v.get("field_path", v.get("field", "unknown"))
        status_raw = str(v.get("status", "not_checked")).lower().strip()

        # Normalize status
        if status_raw in ("verified", "correct", "supported", "confirmed", "true"):
            status = "verified"
        elif status_raw in ("mismatch", "incorrect", "contradicted", "wrong", "false"):
            status = "mismatch"
        elif status_raw in ("unsupported", "not_found", "missing", "absent"):
            status = "unsupported"
        else:
            status = "not_checked"

        evidence_payload = v.get("evidence")
        if isinstance(evidence_payload, list):
            evidence_payload = next(
                (item for item in evidence_payload if isinstance(item, dict)),
                evidence_payload[0] if evidence_payload else None,
            )

        evidence_source = None
        evidence_type = None
        evidence_chunk_id = None
        evidence_start = None
        evidence_end = None
        evidence_score = None
        evidence_excerpt_obj = None
        source_value_obj = None

        if isinstance(evidence_payload, dict):
            evidence_excerpt_obj = (
                evidence_payload.get("excerpt")
                or evidence_payload.get("snippet")
                or evidence_payload.get("detail")
                or evidence_payload.get("reason")
            )
            source_value_obj = (
                evidence_payload.get("source_value")
                or evidence_payload.get("exact_match")
            )
            evidence_source = _to_text(
                evidence_payload.get("source") or evidence_payload.get("source_name")
            )
            evidence_type = _to_text(
                evidence_payload.get("evidence_type") or evidence_payload.get("type")
            )
            evidence_chunk_id = _to_int(evidence_payload.get("chunk_id"))
            evidence_start = _to_int(evidence_payload.get("start"))
            evidence_end = _to_int(evidence_payload.get("end"))
            evidence_score = _to_float(evidence_payload.get("score"))
        else:
            evidence_excerpt_obj = v.get("evidence")

        fv = FieldVerdict(
            status=status,
            field_path=field_path,
            claimed_value=_to_text(v.get("claimed_value") or v.get("value", "")),
            source_value=_to_text(v.get("source_value") or v.get("source") or source_value_obj),
            evidence_excerpt=_to_text(v.get("detail") or evidence_excerpt_obj or v.get("reason")),
            evidence_source=evidence_source,
            evidence_type=evidence_type,
            evidence_chunk_id=evidence_chunk_id,
            evidence_start=evidence_start,
            evidence_end=evidence_end,
            evidence_score=evidence_score,
            severity="critical" if status == "mismatch" else "info",
        )
        field_verdicts.append(fv)

        if status in ("mismatch", "unsupported"):
            issues.append(Issue(
                type=f"audit_{status}",
                field=field_path,
                detail=v.get("detail", f"Audit found {status} for field {field_path}"),
                severity="critical" if status == "mismatch" else "warning",
            ))

    # Parse omissions
    omissions = report.get("omissions", [])
    if isinstance(omissions, str):
        omissions = [omissions]

    for omission in omissions:
        if omission and isinstance(omission, str):
            issues.append(Issue(
                type="omission",
                field="source",
                detail=omission,
                severity="warning",
            ))

    return issues, field_verdicts


def run_claim_audit(
    claim: str,
    source_text: str,
) -> tuple[list[Issue], list[FieldVerdict]]:
    """Run RLM-based audit for a single claim against source text.

    Parameters
    ----------
    claim:
        The claim to verify.
    source_text:
        The source document text.

    Returns
    -------
    tuple[list[Issue], list[FieldVerdict]]
        Issues and field verdicts from the audit.
    """
    import dspy

    chunks = _chunk_source_text(source_text)
    source_manifest = _build_source_manifest(source_text, chunks)

    search_source = _make_search_source(source_text)
    list_source_chunks = _make_list_source_chunks(chunks)
    get_source_chunk = _make_get_source_chunk(chunks)
    search_source_chunks = _make_search_source_chunks(chunks)
    search_numbers = _make_search_numbers(source_text)

    tools = [
        dspy.Tool(
            search_source,
            name="search_source",
            desc="Search the source document for a keyword/phrase. Returns matching snippets with context.",
        ),
        dspy.Tool(
            list_source_chunks,
            name="list_source_chunks",
            desc="List chunk ids and ranges for the source document.",
        ),
        dspy.Tool(
            get_source_chunk,
            name="get_source_chunk",
            desc="Fetch full text for a given chunk id.",
        ),
        dspy.Tool(
            search_source_chunks,
            name="search_source_chunks",
            desc="Search across chunked source text and return chunk-aware matches.",
        ),
        dspy.Tool(
            search_numbers,
            name="search_numbers",
            desc="Find all numbers/measurements in the source document with context.",
        ),
    ]

    class AuditClaim(dspy.Signature):
        """You are a medical document verification auditor.
        Verify whether the claim is supported by the source document.

        Steps:
        1. Use list_source_chunks() to ensure full-source coverage.
        2. Break the claim into individual facts.
        3. For each fact, use search_source_chunks()/get_source_chunk() and search_source() for evidence.
        4. Use search_numbers() to cross-reference numeric values.
        5. Return a JSON audit report.

        audit_report MUST be a JSON object:
        {"field_verdicts": [{"field_path": "claim", "status": "verified|mismatch|unsupported",
        "source_value": "value from source or null",
        "detail": "brief explanation",
        "evidence": {"excerpt": "...", "chunk_id": 12, "start": 420, "end": 460, "score": 7.0}}],
        "summary": "one sentence overall assessment"}
        """

        source_manifest: str = dspy.InputField(desc="Chunk manifest for source navigation")
        claim: str = dspy.InputField(desc="Claim to verify against source")
        audit_report: str = dspy.OutputField(desc="JSON audit report")

    rlm = dspy.RLM(
        AuditClaim,
        max_iterations=15,
        tools=tools,
    )

    try:
        prediction = rlm(source_manifest=source_manifest, claim=claim)
        return _parse_audit_report(prediction.audit_report)
    except Exception as exc:
        if not _should_attempt_outlines_recovery(exc):
            raise
        logger.warning("DSPy RLM claim audit failed; attempting Outlines recovery: %s", exc)
        recovered = _recover_claim_audit_with_outlines(
            claim=claim,
            source_text=source_text,
            source_manifest=source_manifest,
            chunks=chunks,
        )
        if recovered is None:
            raise
        issues, verdicts = _parse_audit_report(json.dumps(recovered, ensure_ascii=False))
        issues.append(
            Issue(
                type="audit_structured_recovery",
                field="claim",
                detail=(
                    "Recovered claim audit output with Outlines constrained JSON generation "
                    f"after DSPy RLM serialization failure: {exc}"
                ),
                severity="warning",
            )
        )
        return issues, verdicts

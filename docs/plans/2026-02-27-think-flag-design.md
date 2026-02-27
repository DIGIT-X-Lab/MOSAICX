# Design: `--think` Extraction Strategy Flag

**Date:** 2026-02-27
**Status:** Approved
**Scope:** CLI, SDK, MCP, extraction pipeline

## Problem

The extraction pipeline uses a fixed cascade: Outlines first, DSPy fallback.
This is optimal for speed but gives users no control over accuracy vs latency.
A 120B model on M4 Max takes ~40s (Outlines) vs ~2-3min (ChainOfThought).
Users need to choose based on their use case: bulk screening vs audit-grade extraction.

## Solution

Add a `--think` flag with three levels: `fast`, `standard` (default), `deep`.

## Interface

### CLI

```bash
mosaicx extract --document report.pdf --template chest_ct --think fast
mosaicx extract --document report.pdf --template chest_ct                # standard (default)
mosaicx extract --document report.pdf --template chest_ct --think deep

# Batch
mosaicx extract --dir ./reports --template chest_ct --think deep --workers 4
```

### SDK

```python
import mosaicx
result = mosaicx.extract(text=report_text, template="chest_ct", think="deep")
```

### MCP

```json
{"tool": "extract", "arguments": {"document": "report.pdf", "template": "chest_ct", "think": "deep"}}
```

## Extraction Behavior Per Level

### `fast` -- Outlines only, no reasoning

```
1. Plan extraction (if document is long)
2. Outlines constrained generation -> coerce -> validate
3. If Outlines fails: dspy.Predict (no reasoning) -> coerce -> validate
4. Deterministic repair only (no dspy.Refine)
5. Semantic validation
```

- LLM calls: 1 (rarely 2)
- Reasoning tokens: 0
- ~40 sec on 120B M4 Max
- Use case: bulk batch screening, simple reports

### `standard` (default) -- Current cascade, unchanged

```
1. Plan extraction (if document is long)
2. Outlines -> complete? -> accept
3. BestOfN (if uncertain sections)
4. Adjudication (if uncertain sections)
5. DSPy typed -> DSPy adapters -> JSON fallback -> Outlines rescue
6. Repair (deterministic + optional Refine per config)
7. Semantic validation
```

- LLM calls: 1-6
- Reasoning tokens: varies
- ~40 sec to ~3 min
- Use case: most clinical extraction

### `deep` -- Both paths, always reason, pick best

```
1. Plan extraction (always, even for short docs)
2. Outlines constrained generation (baseline candidate)
3. dspy.ChainOfThought (always runs, regardless of Outlines result)
4. Score both candidates with _score_extraction_candidate()
5. Pick the higher-scoring result
6. Full repair (deterministic + dspy.Refine enabled)
7. Semantic validation
```

- LLM calls: 2 (Outlines + ChainOfThought), concurrent if server supports batching
- Reasoning tokens: ~1500-3000
- ~2-3 min on 120B
- Use case: complex reports, research datasets, audit-grade extraction

### Comparison

| | fast | standard | deep |
|---|---|---|---|
| Outlines | Primary | First try | Baseline candidate |
| ChainOfThought | Never | Fallback only | Always |
| dspy.Predict | Fallback | Part of cascade | Not used |
| Planner | If long doc | If long doc | Always |
| dspy.Refine repair | Disabled | Config-dependent | Enabled |
| BestOfN/Adjudication | Skipped | If uncertain | Skipped (score-based pick) |
| Candidate scoring | Not used | Not used | Outlines vs CoT scored |

## Code Threading Path

The `think` parameter is a `Literal["fast", "standard", "deep"]` threaded through:

```
CLI:  extract() in cli.py
  click.option("--think", type=Choice(["fast", "standard", "deep"]), default="standard")
    -> _extract_batch()  (if --dir)
    -> process_fn closure captures think

SDK:  mosaicx.extract(think="deep") in sdk.py
    -> _extract_single_text(think=...)

MCP:  extract tool handler in mcp_server.py
    -> _extract_single_text(think=...)

All paths converge:
  -> DocumentExtractor(output_schema=..., think=...)
     stores self._think
     -> forward() passes to _extract_schema_with_structured_chain(think=...)

Core logic in _extract_schema_with_structured_chain():
  if think == "fast":
      try Outlines -> if fails, dspy.Predict -> coerce -> done
      (skip BestOfN, adjudication, ChainOfThought, Refine)

  elif think == "deep":
      run Outlines (baseline)
      run dspy.ChainOfThought (always)
      score both -> pick best
      repair with Refine enabled -> done

  else:  # standard
      current cascade unchanged (zero behavior change)
```

## Files Changed

| File | Change |
|------|--------|
| `mosaicx/cli.py` | Add `--think` option, thread to `_extract_batch` and extraction calls |
| `mosaicx/sdk.py` | Add `think` param to `extract()` and `_extract_single_text()` |
| `mosaicx/mcp_server.py` | Add `think` param to extract tool handler |
| `mosaicx/report.py` | Add `think` param to `run_report()`, pass to `DocumentExtractor` |
| `mosaicx/pipelines/extraction.py` | Add `think` param to `DocumentExtractor.__init__`, `forward()`, `_extract_schema_with_structured_chain()` |

## What Does NOT Change

- No new config fields (CLI flag only)
- No changes to mode pipelines (radiology, pathology -- they have their own forward())
- No changes to the coercion layer, scoring function, or repair logic
- `standard` is default -- zero breaking changes to existing behavior
- No changes to template loading or schema compilation
- No changes to the planner/router logic (except `deep` forces planner on short docs)

## Performance Expectations (120B-4bit, M4 Max 128 GB)

| Think level | Single doc | Batch 100 docs (1 worker) |
|---|---|---|
| fast | ~40 sec | ~67 min |
| standard | ~40 sec - 3 min | ~67 min - 5 hrs |
| deep | ~2-3 min | ~3.5 - 5 hrs |

## Future Considerations (Not In Scope)

- `--think auto` mode that picks level based on document complexity
- Config/env var default (`MOSAICX_THINK`)
- Per-document think level in batch JSONL input
- Concurrent Outlines + CoT execution in `deep` mode (requires --continuous-batching)

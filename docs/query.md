# Query Guide

Use `mosaicx query` for grounded Q&A across one or more sources with citations.

## What `query` Is For

- Fast investigation of documents without writing extraction templates first.
- Multi-turn analysis with session memory.
- Evidence-backed answers for developer debugging and RAG checks.

## Core Commands

One-shot question:

```bash
mosaicx query --sources "reports/*.pdf" -q "What changed between scans?"
```

Interactive chat:

```bash
mosaicx query --sources "reports/*.pdf" --chat
```

Save output:

```bash
mosaicx query --sources "reports/*.pdf" -q "Summarize progression" -o answer.json

# Runtime preflight (Deno sandbox)
mosaicx runtime check
```

## Key Runtime Controls

- `--max-iterations`: higher is deeper reasoning, slower responses.
- `--citations`: number of evidence rows shown per answer.

Tabular analytics backend:

- Install `mosaicx[query]` to enable high-performance tabular tools (`duckdb`, `polars`).
- Cohort stats (mean/median/count, grouped stats, SQL) are computed deterministically.

Examples:

```bash
mosaicx query --sources "reports/*.pdf" -q "modality used?" --max-iterations 4
mosaicx query --sources "reports/*.pdf" -q "timeline" --max-iterations 8 --citations 5
```

## How to Read Results

Prioritize:

- `answer`: final grounded response.
- `citations`: source snippets used.
- `confidence`: grounding confidence.
- `fallback_used`: LLM failure fallback indicator.
- `rescue_used`: answer was repaired/reconciled with evidence.
- `Evidence Type`:
  - `computed` = deterministic stat/SQL output from table tools
  - `row_match` = matching table rows
  - `text_match` = matching document text

Evidence display order in CLI:

- `Computed` block appears first for tabular statistics (includes execution engine).
- `Supporting` block appears after that for row/text snippets.

## Recommended Developer Workflow

1. Ask the question in `query`.
2. Inspect citations and confidence.
3. If confidence is low, ask narrower follow-ups.
4. For hard true/false adjudication, run `verify` on the final claim.

## Performance Tips

- Use smaller `--max-iterations` for quick triage.
- Increase `--max-iterations` for cross-document reasoning.
- Keep sources scoped to relevant files to reduce latency.

## Runtime Notes

- `query` uses DSPy RLM with a Deno-backed sandbox for tool/code execution.
- On first run, MOSAICX can prompt to install Deno if missing.
- In automation/CI, set `MOSAICX_AUTO_INSTALL_RUNTIME=1` to allow non-interactive runtime install attempts.

## Common Pitfalls

- Treating `query` answers as binary truth without running `verify`.
- Ignoring low-confidence responses even when answer text looks plausible.
- Using too broad source globs for focused clinical questions.

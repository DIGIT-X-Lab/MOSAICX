# Troubleshooting

Use this page when MOSAICX behavior is slow, inaccurate, or inconsistent.

## Query Is Slow on CSV/Parquet

Symptoms:

- `mosaicx query` takes a long time to answer.
- You see `RLM reached max iterations`.

What to do:

1. Install the tabular query stack if not already installed:

```bash
pip install 'mosaicx[query]'
```

2. Lower iteration budget for interactive use:

```bash
mosaicx query --sources "data/*.csv" --chat --max-iterations 4
```

3. Scope sources to the files you actually need.
4. Ask narrower questions first, then follow up.
5. Confirm model server health:

```bash
curl -sS --max-time 5 http://127.0.0.1:8000/v1/models
```

## Query Answer Is Wrong for Cohort Statistics

Symptoms:

- Mean/median/count answers are wrong for tabular data.
- Evidence shows only headers or generic snippets.

What to do:

1. Use a direct statistical phrasing:
   - `what is the average BMI?`
   - `what is the median age?`
   - `how many unique Scanner values are there?`
2. Increase citations so you can inspect computed evidence:

```bash
mosaicx query --sources cohort.csv -q "average bmi of cohort" --citations 5
```

3. If the answer still looks wrong, run a second query with explicit column naming.
4. For binary adjudication, verify the final claim with `mosaicx verify`.

## Evidence Looks Unclear

Current evidence types:

- `computed`: table statistic computed from the data.
- `row_match`: matching table rows.
- `text_match`: matching text snippets.

If you only see `text_match` for table questions, the source may not be loaded as a dataframe (e.g., malformed CSV). Verify source catalog in query startup output.

## Verify Falls Back from Thorough/Standard

Symptoms:

- Requested `--level thorough` but effective level is lower.

What to check:

1. `effective_level` in JSON output.
2. `fallback_used` and `fallback_reason`.
3. LLM server availability and credentials.

## Deno Warning in Query

Symptom:

- `WARNING dspy.primitives.python_interpreter: Unable to find the Deno cache dir`

Fix:

1. Ensure Deno is installed and on `PATH`.
2. Set `DENO_DIR` to a writable directory if needed.
3. Restart your shell after installation.

## OCR Quality Warnings

Symptoms:

- `Low OCR quality detected ... results may be unreliable`

What to do:

1. Increase scan quality/DPI.
2. Force OCR if native text extraction is poor:

```bash
export MOSAICX_FORCE_OCR=true
```

3. Tune OCR engine in config (`surya`, `chandra`, `both`).

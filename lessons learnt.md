# Lessons Learnt

## Why this matters

For a structured extraction SDK, OCR quality is not a minor preprocessing detail. It is one of the main determinants of extraction quality, grounding quality, and downstream reliability.

This investigation used the same lab report across multiple paths:

- `mosaicx extract` on the original PDF with `--force-ocr`
- PPStructure OCR output
- Chandra OCR output
- `mosaicx extract` on the Chandra-generated Markdown

## Core lessons

### 1. OCR quality strongly affects extraction quality

The structured extractor often succeeded on the first attempt in both runs, but the field quality still changed materially based on OCR quality.

Examples from the runs:

- PPStructure-based PDF extraction produced `patient_name = "Mrs SAKUNTHALA S62Y/F"`
- Chandra Markdown-based extraction produced `patient_name = "Mrs SAKUNTHALA S"`
- PPStructure merged timestamps and labels like `28-03-202612:27:17`
- Chandra separated dates and times cleanly
- PPStructure had noisier table text
- Chandra reconstructed the hematology table much more cleanly

Conclusion:

If OCR is noisy, the extractor can still "succeed" while returning lower-quality structured output.

### 2. First-attempt structured success does not mean the pipeline is healthy

In the original PDF run, the structured path was:

- `outlines_fast`
- no structured fallback

That sounds good, but the quality issues were already baked into the OCR text before structured extraction began.

Conclusion:

Pipeline success metrics must separate:

- OCR quality
- extraction success
- grounding quality
- final field correctness

### 3. Cleaner text reduces the need for orchestration tricks

At the time of the first PPStructure run, the extraction hot path still used a long-document ReAct planner. That has since been removed from the default runtime.

On the first PPStructure OCR run:

- OCR text was `4170` chars
- planner used `react`
- text was compressed to `2009` chars before extraction

On the Chandra Markdown run:

- document was `2550` chars
- planner used `short_doc_bypass`
- no ReAct compression was applied

Conclusion:

When the input text is clean and compact, the system can stay simpler and safer. Better OCR reduces pressure to use risky preprocessing heuristics.

Current state after cleanup:

- long documents now use `planner = "full_text_default"`
- route `heavy_extract`
- `react_used = false`
- full OCR text is preserved in the extraction hot path

### 4. Document completeness matters as much as OCR quality

The Chandra Markdown extraction looked cleaner, but it only represented one page of the PDF, not all four pages.

That happened because the Chandra CLI was run with `--page-range 1`, and its page-range handling is off by one for PDFs. It treated `1` as internal page index `1`, which is the second PDF page.

Impact:

- Chandra extraction returned `sample_type = "EDTA BLOOD"`
- full PDF extraction returned `sample_type = "SERUM"`

Both were "correct" for the page they came from, but only one represented the full document context.

Conclusion:

A cleaner OCR result from the wrong subset of pages is still an unreliable input for extraction.

### 5. Provenance and debug traces are essential

Useful findings came from the internal trace, not just the final JSON:

- planner path
- structured chain
- whether fallback was used
- repair stage behavior
- grounded excerpts

Conclusion:

For developer-grade extraction, these traces should be first-class outputs, not hidden behind special env flags.

### 6. OCR engine integration quality matters, not just the OCR model

Chandra itself produced high-quality output locally once HF dependencies were installed, but the integration had operational issues:

- local HF required extra dependencies
- `MOSAICX_CHANDRA_SERVER_URL` is currently not wired into the loader path
- pointing Chandra vLLM mode at a text-only model endpoint produced empty OCR
- Chandra CLI page-range behavior for PDFs is off by one

Conclusion:

An OCR engine is only useful if the runtime and integration are dependable.

## What the runs suggest

### PPStructure on the full PDF

Strengths:

- good enough to drive extraction
- handled all pages
- preserved useful table structure

Weaknesses:

- merged tokens
- noisier demographics and dates
- poorer grounding excerpts
- planner compression was triggered on the OCR text

### Chandra on one hematology page

Strengths:

- much cleaner text
- much better table reconstruction
- better separation of patient/demographic/timestamp information

Weaknesses:

- only one page was processed in this test
- local HF inference is heavy and slow
- integration/runtime issues still need cleanup

## Product-level implications

If the goal is a world-class extraction engine for developers, especially for tumor registrar replacement workflows, then:

1. OCR should be treated as a primary product surface, not a hidden preprocessing step.
2. Extraction benchmarking should always include OCR provenance.
3. Full-document coverage checks are mandatory.
4. Planner and fallback complexity should not be used to compensate for weak OCR.
5. The best architecture is likely:
   one strong OCR layer, one clear structured extraction layer, one explicit validation layer, and full traceability.

## Practical takeaway

Better OCR does not guarantee correct extraction, but weak OCR almost guarantees lower extraction quality.

In this codebase, OCR quality is a first-order factor, not a secondary optimization.

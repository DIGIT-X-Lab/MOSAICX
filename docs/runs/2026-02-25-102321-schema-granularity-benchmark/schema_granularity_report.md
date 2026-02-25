# Schema Granularity Benchmark

- Generated: 2026-02-25 10:40:55Z
- Cases: `/Users/nutellabear/Documents/00-Code/MOSAICX/tests/datasets/evaluation/schema_granularity_cases.json`
- Modes: `baseline` (semantic gate off) vs `hybrid` (semantic gate on + DSPy assessor)
- Generation context: `describe_only`

## Aggregate

| Metric | Baseline | Hybrid | Delta |
|---|---:|---:|---:|
| semantic_score_mean | 0.5926 | 0.5913 | -0.0013 |
| required_coverage_mean | 0.9000 | 0.9333 | +0.0333 |
| extraction_success_rate | 1.0000 | 1.0000 | +0.0000 |
| repeated_structure_pass_rate | 1.0000 | 1.0000 | +0.0000 |
| enum_pass_rate | 1.0000 | 1.0000 | +0.0000 |
| mean_list_object_fields | 0.8000 | 0.8000 | +0.0000 |
| mean_enum_fields | 2.2000 | 2.0000 | -0.2000 |
| semantic_gate_trigger_rate | 0.0000 | 0.0000 | +0.0000 |

## Per Case

| Case | Mode | Semantic | ReqCov | ExtractOK | RepeatOK | EnumOK | ListObj | EnumFields | Gen(s) | Extract(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cspine_pdf | baseline | 0.6917 | 1.0000 | yes | yes | yes | 1 | 1 | 27.99 | 25.77 |
| cspine_pdf | hybrid | 0.6917 | 1.0000 | yes | yes | yes | 1 | 1 | 31.99 | 34.36 |
| oncology_lesions | baseline | 0.5611 | 1.0000 | yes | yes | yes | 1 | 6 | 58.18 | 116.14 |
| oncology_lesions | hybrid | 0.5464 | 1.0000 | yes | yes | yes | 1 | 4 | 75.89 | 73.04 |
| pathology_biomarkers | baseline | 0.5600 | 0.5000 | yes | yes | yes | 1 | 3 | 101.26 | 44.04 |
| pathology_biomarkers | hybrid | 0.5409 | 0.6667 | yes | yes | yes | 1 | 3 | 143.97 | 27.53 |
| medication_list | baseline | 0.6125 | 1.0000 | yes | yes | yes | 1 | 1 | 80.96 | 10.90 |
| medication_list | hybrid | 0.6400 | 1.0000 | yes | yes | yes | 1 | 2 | 72.78 | 14.59 |
| simple_vitals_note | baseline | 0.5375 | 1.0000 | yes | yes | yes | 0 | 0 | 44.75 | 13.01 |
| simple_vitals_note | hybrid | 0.5375 | 1.0000 | yes | yes | yes | 0 | 0 | 33.57 | 21.82 |

## Notes

- Benchmarks use the configured local DSPy LM and adapter policy.
- DSPy cache should be cleared before this run for cold-start comparability.

# Schema Granularity Benchmark

- Generated: 2026-02-25 10:14:45Z
- Cases: `/Users/nutellabear/Documents/00-Code/MOSAICX/tests/datasets/evaluation/schema_granularity_cases.json`
- Modes: `baseline` (semantic gate off) vs `hybrid` (semantic gate on + DSPy assessor)

## Aggregate

| Metric | Baseline | Hybrid | Delta |
|---|---:|---:|---:|
| semantic_score_mean | 0.5604 | 0.5931 | +0.0327 |
| required_coverage_mean | 1.0000 | 1.0000 | +0.0000 |
| extraction_success_rate | 1.0000 | 1.0000 | +0.0000 |
| repeated_structure_pass_rate | 1.0000 | 1.0000 | +0.0000 |
| enum_pass_rate | 1.0000 | 1.0000 | +0.0000 |
| mean_list_object_fields | 0.8000 | 0.8000 | +0.0000 |
| mean_enum_fields | 1.0000 | 1.8000 | +0.8000 |
| semantic_gate_trigger_rate | 0.0000 | 0.0000 | +0.0000 |

## Per Case

| Case | Mode | Semantic | ReqCov | ExtractOK | RepeatOK | EnumOK | ListObj | EnumFields | Gen(s) | Extract(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cspine_pdf | baseline | 0.5261 | 1.0000 | yes | yes | yes | 1 | 1 | 66.34 | 84.53 |
| cspine_pdf | hybrid | 0.5091 | 1.0000 | yes | yes | yes | 1 | 4 | 279.91 | 45.40 |
| oncology_lesions | baseline | 0.6091 | 1.0000 | yes | yes | yes | 1 | 3 | 58.25 | 23.97 |
| oncology_lesions | hybrid | 0.6562 | 1.0000 | yes | yes | yes | 1 | 2 | 42.04 | 19.98 |
| pathology_biomarkers | baseline | 0.5667 | 1.0000 | yes | yes | yes | 1 | 1 | 83.26 | 21.14 |
| pathology_biomarkers | hybrid | 0.5667 | 1.0000 | yes | yes | yes | 1 | 1 | 114.11 | 17.11 |
| medication_list | baseline | 0.5000 | 1.0000 | yes | yes | yes | 1 | 0 | 80.51 | 13.78 |
| medication_list | hybrid | 0.6333 | 1.0000 | yes | yes | yes | 1 | 2 | 70.78 | 12.01 |
| simple_vitals_note | baseline | 0.6000 | 1.0000 | yes | yes | yes | 0 | 0 | 22.23 | 7.31 |
| simple_vitals_note | hybrid | 0.6000 | 1.0000 | yes | yes | yes | 0 | 0 | 23.89 | 7.24 |

## Notes

- Benchmarks use the configured local DSPy LM and adapter policy.
- DSPy cache should be cleared before this run for cold-start comparability.

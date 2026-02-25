# Schema Granularity Benchmark

- Generated: 2026-02-25 08:47:36Z
- Cases: `/Users/nutellabear/Documents/00-Code/MOSAICX/tests/datasets/evaluation/schema_granularity_cases.json`
- Modes: `baseline` (semantic gate off) vs `hybrid` (semantic gate on + DSPy assessor)

## Aggregate

| Metric | Baseline | Hybrid | Delta |
|---|---:|---:|---:|
| semantic_score_mean | 0.5872 | 0.5759 | -0.0113 |
| required_coverage_mean | 1.0000 | 0.9833 | -0.0167 |
| extraction_success_rate | 1.0000 | 1.0000 | +0.0000 |
| repeated_structure_pass_rate | 1.0000 | 1.0000 | +0.0000 |
| enum_pass_rate | 1.0000 | 1.0000 | +0.0000 |
| mean_list_object_fields | 0.8000 | 0.8000 | +0.0000 |
| mean_enum_fields | 1.8000 | 2.6000 | +0.8000 |
| semantic_gate_trigger_rate | 0.0000 | 0.2000 | +0.2000 |

## Per Case

| Case | Mode | Semantic | ReqCov | ExtractOK | RepeatOK | EnumOK | ListObj | EnumFields | Gen(s) | Extract(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cspine_pdf | baseline | 0.5333 | 1.0000 | yes | yes | yes | 1 | 3 | 110.77 | 68.92 |
| cspine_pdf | hybrid | 0.5380 | 0.9167 | yes | yes | yes | 1 | 7 | 528.00 | 71.12 |
| oncology_lesions | baseline | 0.6091 | 1.0000 | yes | yes | yes | 1 | 3 | 156.87 | 24.63 |
| oncology_lesions | hybrid | 0.6500 | 1.0000 | yes | yes | yes | 1 | 4 | 58.81 | 24.87 |
| pathology_biomarkers | baseline | 0.5667 | 1.0000 | yes | yes | yes | 1 | 1 | 86.51 | 12.52 |
| pathology_biomarkers | hybrid | 0.5667 | 1.0000 | yes | yes | yes | 1 | 1 | 117.12 | 13.22 |
| medication_list | baseline | 0.6333 | 1.0000 | yes | yes | yes | 1 | 2 | 75.73 | 10.15 |
| medication_list | hybrid | 0.5250 | 1.0000 | yes | yes | yes | 1 | 1 | 109.44 | 10.17 |
| simple_vitals_note | baseline | 0.5938 | 1.0000 | yes | yes | yes | 0 | 0 | 41.42 | 8.23 |
| simple_vitals_note | hybrid | 0.6000 | 1.0000 | yes | yes | yes | 0 | 0 | 23.91 | 6.13 |

## Notes

- Benchmarks use the configured local DSPy LM and adapter policy.
- DSPy cache should be cleared before this run for cold-start comparability.

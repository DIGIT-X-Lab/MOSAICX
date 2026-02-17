# MOSAICX Examples

Hands-on examples and sample data to get you productive with MOSAICX fast.

## Directory Layout

```
examples/
├── quickstart/                  # Tutorial scripts — run these first
│   ├── 01_basic_extraction.sh   # Your first extraction
│   ├── 02_schema_workflow.sh    # Create and use schemas
│   ├── 03_batch_processing.sh   # Process many documents at once
│   ├── 04_optimization.sh       # Tune pipeline accuracy with DSPy
│   └── 05_deidentification.sh   # Remove PHI from clinical notes
│
├── data/
│   ├── reports/                 # Sample medical documents (synthetic)
│   │   ├── radiology_ct_chest.txt
│   │   ├── radiology_mri_brain.txt
│   │   ├── pathology_breast_biopsy.txt
│   │   ├── pathology_colon_resection.txt
│   │   ├── clinical_note_admission.txt
│   │   └── echo_report.txt
│   │
│   ├── training/                # Labeled JSONL datasets for optimization
│   │   ├── radiology_train.jsonl    (10 examples)
│   │   ├── radiology_test.jsonl     (5 examples)
│   │   ├── pathology_train.jsonl    (8 examples)
│   │   ├── pathology_test.jsonl     (4 examples)
│   │   ├── extract_train.jsonl      (8 examples)
│   │   └── extract_test.jsonl       (4 examples)
│   │
│   └── outputs/                 # What MOSAICX output looks like
│       ├── radiology_mode_output.json
│       ├── pathology_mode_output.json
│       ├── auto_extraction_output.json
│       ├── schema_extraction_output.json
│       ├── deidentify_output.json
│       └── batch_summary.json
```

## Where to Start

**Never used MOSAICX before?** Run the first tutorial:

```bash
cd examples/quickstart
bash 01_basic_extraction.sh
```

**Want to see what the output looks like before running anything?** Browse `data/outputs/`.

**Want to try optimization?** The training datasets are ready to go:

```bash
mosaicx optimize --pipeline radiology \
  --trainset examples/data/training/radiology_train.jsonl \
  --budget light
```

## Tutorials

| Script | What You'll Learn | Time |
|--------|-------------------|------|
| `01_basic_extraction.sh` | Run extractions in auto and radiology mode | 5 min |
| `02_schema_workflow.sh` | Generate, refine, and use custom schemas | 10 min |
| `03_batch_processing.sh` | Process a directory of documents in parallel | 5 min |
| `04_optimization.sh` | Tune accuracy with labeled data and DSPy | 15 min |
| `05_deidentification.sh` | Strip PHI from clinical notes | 5 min |

Each script is self-contained, well-commented, and uses the sample data in `data/reports/`.

## Sample Data

All medical data in this directory is **completely synthetic**. No real patient information is included. The reports are designed to be realistic enough to produce meaningful extraction results.

### Reports

Six synthetic medical documents covering different specialties:

- **Radiology:** CT chest, MRI brain
- **Pathology:** Breast biopsy, colon resection
- **Clinical:** Admission note with fake PHI (for testing de-identification)
- **Cardiology:** Echocardiography report

### Training Datasets

Pre-labeled JSONL files for DSPy optimization. Each record contains input fields and gold-standard labels:

| Dataset | Pipeline | Train | Test |
|---------|----------|-------|------|
| Radiology | `radiology` | 10 examples | 5 examples |
| Pathology | `pathology` | 8 examples | 4 examples |
| Extraction | `extract` | 8 examples | 4 examples |

Use these with `mosaicx optimize` and `mosaicx eval`. See [Optimization Guide](../docs/optimization.md) for details.

### Example Outputs

Pre-generated JSON files showing what MOSAICX produces for each pipeline. Browse these to understand the output structure before running any commands.

## Full Documentation

For comprehensive guides, see the [docs/](../docs/README.md) directory.

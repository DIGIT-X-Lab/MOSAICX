# Tutorial 1: Creating an Extraction Template

Templates tell MOSAICX what to extract from a document. You describe the fields you want, and the LLM fills them in.

## Quick Start

Create a template by describing what you want:

```bash
mosaicx template create --describe "patient demographics: name, ID, date of birth, sex, BMI"
```

The LLM generates a YAML template and saves it to `~/.mosaicx/templates/`.

## From a Sample Document

If you have a sample document, the LLM can infer the template from it:

```bash
mosaicx template create --document sample_report.pdf --describe "extract all patient vitals and demographics"
```

## List Available Templates

```bash
mosaicx template list
```

Shows both built-in templates (chest_ct, brain_mri, etc.) and your custom templates.

## Inspect a Template

```bash
mosaicx template show BasicExtraction
```

Shows the fields, types, descriptions, and which are required.

## Template YAML Format

Templates are YAML files in `~/.mosaicx/templates/`. Here's a simple one:

```yaml
name: PatientVitals
description: Extract patient demographics and vital signs from a clinical document.
sections:
  - name: patient_name
    type: str
    required: true
    description: Full name of the patient.

  - name: patient_id
    type: str
    required: true
    description: Unique patient identifier (e.g. MRN, PID).

  - name: date_of_birth
    type: str
    required: true
    description: Date of birth (YYYY-MM-DD format).

  - name: sex
    type: str
    required: true
    description: Patient sex.
    enum: [Male, Female, Other]

  - name: vitals
    type: list
    required: false
    description: List of vital sign measurements.
    item:
      type: object
      fields:
        - name: vital_sign
          type: str
          description: Name of the vital sign (e.g. Blood Pressure, Heart Rate).
        - name: value
          type: str
          description: Measured value with units.
        - name: normal_range
          type: str
          required: false
          description: Normal reference range.
        - name: status
          type: str
          required: false
          description: Clinical status (e.g. Normal, Elevated).
```

## Refine an Existing Template

Made a template but want to tweak it? Use refine:

```bash
mosaicx template refine PatientVitals --instruction "add a field for attending physician name"
```

## Validate a Template

Check that a template compiles correctly before using it:

```bash
mosaicx template validate ~/.mosaicx/templates/PatientVitals.yaml
```

## What's Next

Once you have a template, use it for extraction:

```bash
mosaicx extract --document report.pdf --template PatientVitals -o result.json
```

See [Tutorial 2: Extracting Data](02-extraction.md).

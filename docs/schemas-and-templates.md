# Schemas and Templates

This guide explains how to create, manage, and use templates in MOSAICX. Templates are YAML files that define exactly what data gets extracted from a clinical document and how it is structured. Whether you are extracting patient vitals from a clinical note or structured findings from a radiology report, a template gives you precise control over the output.

## What is a Template?

A template defines WHAT you want to extract from a document. Instead of letting the LLM decide what to extract, you tell it exactly which fields to look for, what types they should be, and how they nest together.

Think of a template as a form. When you read a radiology report, you want to fill out specific sections: indication (text), nodules (a list, each with location, size, and morphology), and impression (text). The template tells the LLM exactly what information to find and how to structure it.

Templates are written in YAML and live in one of three places:

- **Built-in templates** -- 11 templates ship with MOSAICX, covering common radiology exams.
- **User templates** -- Templates you create, stored in `~/.mosaicx/templates/`.
- **Project templates** -- YAML files anywhere on disk, referenced by path.

## YAML Template Format

Every template is a YAML file with the following top-level keys:

```yaml
name: ChestCTReport
description: "Structured chest CT radiology report"
mode: radiology                # optional -- pipeline mode to use
radreport_id: "RDES3"         # optional -- RadReport standard ID
sections:
  - name: indication
    type: str
    required: true
    description: "Clinical indication for exam"
  - name: nodules
    type: list
    required: false
    description: "Pulmonary nodules"
    item:
      type: object
      fields:
        - name: location
          type: str
          description: "Anatomic location (lobe, segment)"
        - name: size_mm
          type: float
          description: "Maximum diameter in millimeters"
        - name: lung_rads
          type: str
          required: false
          description: "Lung-RADS category"
  - name: impression
    type: str
    required: true
    description: "Summary impression with recommendations"
```

### Top-level Keys

| Key | Required | Description |
|-----|----------|-------------|
| `name` | yes | Template name (used as the generated Pydantic class name) |
| `description` | no | Human-readable description of what this template captures |
| `mode` | no | Pipeline mode to use for extraction (e.g. `radiology`, `pathology`) |
| `radreport_id` | no | RadReport.org standard ID, for built-in templates |
| `sections` | yes | List of section definitions (the fields to extract) |

### Section Definition

Each entry in `sections` is a field definition:

| Key | Required | Description |
|-----|----------|-------------|
| `name` | yes | Field identifier (must be a valid Python variable name) |
| `type` | yes | Data type (see below) |
| `required` | no | Whether the field must be present in output (default: `true` for top-level, inherits for nested) |
| `description` | no | Human-readable explanation -- the LLM uses this to understand what to extract |
| `values` | enum only | List of allowed values for `enum` type |
| `item` | list only | Definition of list element type |
| `fields` | object only | List of nested field definitions |

### Supported Types

| Type | Description | Example |
|------|-------------|---------|
| `str` | Text string | `"Left upper lobe consolidation"` |
| `int` | Integer number | `72` |
| `float` | Decimal number | `5.3` |
| `bool` | True/false | `true` |
| `enum` | One of a fixed set of values | Requires `values` list |
| `list` | List of items | Requires `item` definition |
| `object` | Nested structure with named fields | Requires `fields` list |

### Nesting Examples

**Enum field:**

```yaml
- name: breast_composition
  type: enum
  required: true
  description: "ACR breast density category"
  values: ["a", "b", "c", "d"]
```

**List of objects:**

```yaml
- name: lesions
  type: list
  required: false
  description: "FDG-avid lesions"
  item:
    type: object
    fields:
      - name: location
        type: str
        description: "Anatomic location"
      - name: suv_max
        type: float
        description: "Maximum standardized uptake value"
      - name: size_mm
        type: float
        required: false
        description: "Size in millimeters on CT"
```

**Deeply nested structures** (list of objects containing a nested list):

```yaml
- name: findings
  type: list
  item:
    type: object
    fields:
      - name: anatomy
        type: str
      - name: measurements
        type: list
        item:
          type: object
          fields:
            - name: dimension
              type: str
            - name: value
              type: float
            - name: unit
              type: str
```

## Template Resolution

When you pass `--template` to `extract` or `batch`, MOSAICX resolves the value through a chain of lookups. Understanding this chain helps you organize your templates.

**Resolution order:**

1. **File path** -- If the value ends in `.yaml` or `.yml` and the file exists on disk, it is compiled directly.
2. **User templates** -- Looks for `~/.mosaicx/templates/<name>.yaml`.
3. **Built-in templates** -- Looks in the package `schemas/radreport/templates/<name>.yaml` directory (the 11 built-in templates).
4. **Legacy saved schema** -- Looks for `~/.mosaicx/schemas/<name>.json` (backward compatibility with older JSON schemas).
5. **Error** -- If none of the above match, an error is raised with suggestions.

This means you can override a built-in template by placing a file with the same name in `~/.mosaicx/templates/`. Your user template will take precedence.

**Examples of how different values resolve:**

```bash
# 1. File path -- compiles the YAML file directly
mosaicx extract --document scan.pdf --template ./my_templates/echo.yaml

# 2. User template -- looks up ~/.mosaicx/templates/echo.yaml
mosaicx extract --document scan.pdf --template echo

# 3. Built-in template -- uses the packaged chest_ct.yaml
mosaicx extract --document scan.pdf --template chest_ct

# 4. Legacy schema -- loads ~/.mosaicx/schemas/EchoReport.json
mosaicx extract --document scan.pdf --template EchoReport
```

## Built-in Templates

MOSAICX ships with 11 built-in templates for common radiology exams. These are based on RadReport standards and can be used directly or as starting points for your own templates.

| Name | Mode | RDES ID | Description |
|------|------|---------|-------------|
| `generic` | radiology | -- | Generic radiology report (indication, comparison, technique, findings, impression) |
| `chest_ct` | radiology | RDES3 | Chest CT with lung parenchyma, nodules, airways, mediastinum, pleura, heart/vessels |
| `chest_xr` | radiology | RDES2 | Chest X-ray report |
| `brain_mri` | radiology | RDES28 | Brain MRI with parenchyma, ventricles, extra-axial spaces, posterior fossa, vessels |
| `abdomen_ct` | radiology | RDES44 | Abdomen CT report |
| `mammography` | radiology | RDES4 | Mammography with breast composition, masses, calcifications, BI-RADS |
| `thyroid_us` | radiology | RDES72 | Thyroid ultrasound report |
| `lung_ct` | radiology | RDES195 | Lung CT screening report |
| `msk_mri` | radiology | -- | MSK MRI report |
| `cardiac_mri` | radiology | RDES214 | Cardiac MRI report |
| `pet_ct` | radiology | RDES76 | PET/CT with FDG-avid lesions, SUV measurements, response assessment |

To view the full structure of any built-in template:

```bash
mosaicx template show chest_ct
```

To list all available templates (built-in and user):

```bash
mosaicx template list
```

## Creating Templates

There are several ways to create a template: from a description, from a sample document, from a URL, from a RadReport ID, or by converting a legacy JSON schema.

### From a Description

Describe what you want in plain English and the LLM generates a YAML template:

```bash
mosaicx template create --describe "echocardiography report with LVEF, valve grades, wall motion, and clinical impression"
```

Optionally name your template and set a pipeline mode:

```bash
mosaicx template create \
  --describe "echocardiography report with LVEF, valve grades, and clinical impression" \
  --name EchoReport \
  --mode radiology
```

### From a Sample Document

If you have a sample report, the LLM can read it and infer the template structure:

```bash
mosaicx template create --from-document sample_report.pdf
```

You can combine a document with a description for more control:

```bash
mosaicx template create \
  --from-document sample_report.pdf \
  --describe "focus on cardiac measurements and valve assessments"
```

### From a URL

Generate a template from web page content (useful for RadReport.org pages and similar references):

```bash
mosaicx template create --from-url https://radreport.org/home/50
```

### From a RadReport ID

Fetch a RadReport template directly by ID and enrich it with LLM-inferred types:

```bash
mosaicx template create --from-radreport RPT50890
```

### From a Legacy JSON Schema

Convert an existing JSON SchemaSpec file to a YAML template:

```bash
mosaicx template create --from-json ~/.mosaicx/schemas/EchoReport.json
```

### Save to a Custom Location

By default, templates are saved to `~/.mosaicx/templates/`. You can save to a different location:

```bash
mosaicx template create \
  --describe "vital signs" \
  --output ./my_project/templates/vitals.yaml
```

## Managing Templates

### List All Templates

See all available templates (built-in and user-created):

```bash
mosaicx template list
```

### View a Template

See the detailed structure of a template:

```bash
mosaicx template show chest_ct
```

**Example output:**

```
ChestCTReport (built-in)

Structured chest CT radiology report (RDES3)

  Mode: radiology
  RDES: RDES3
  Source: .../schemas/radreport/templates/chest_ct.yaml

  Section         Type           Req  Description
  indication      str            yes  Clinical indication for chest CT
  comparison      str            --   Prior comparison studies and dates
  technique       str            --   CT technique, contrast, slice thickness
  lungs           str            yes  Lung parenchyma findings
  nodules         list[object]   --   Pulmonary nodules
  airways         str            --   Trachea and bronchial tree findings
  mediastinum     str            --   Mediastinal and hilar findings
  lymph_nodes     str            --   Lymph node stations and measurements
  pleura          str            --   Pleural findings including effusions
  heart_and_vessels str          --   Cardiac and vascular findings
  chest_wall      str            --   Chest wall and osseous findings
  upper_abdomen   str            --   Included upper abdominal findings
  impression      str            yes  Summary impression with recommendations

  13 sections
```

### Validate a Template

Check if your YAML template is valid before using it:

```bash
mosaicx template validate --file ./templates/echo.yaml
```

If there are errors (invalid field types, missing required attributes, etc.), you get a detailed error message.

## Refining Templates

After creating a template, you can modify it using LLM-powered natural language instructions:

```bash
mosaicx template refine EchoReport \
  --instruction "add a field for pericardial effusion severity as an enum with values none, trivial, small, moderate, large"
```

The LLM updates the template according to your instruction. This is useful for:

- Adding or removing fields
- Changing field types
- Adding complex nested structures
- Reorganizing the template

When refining a built-in template, the result is saved as a user template (the built-in is never modified). When refining a user template, the previous version is automatically archived.

```bash
mosaicx template refine chest_ct \
  --instruction "add a covid_findings section for ground-glass opacities and consolidation patterns"
```

You can also save the refined template to a custom path:

```bash
mosaicx template refine chest_ct \
  --instruction "simplify to just indication, findings, and impression" \
  --output ./templates/chest_ct_simple.yaml
```

## Version History

Every time you refine a user template, MOSAICX automatically archives the previous version in `~/.mosaicx/templates/.history/`. This means you can always roll back if needed.

### View History

See all archived versions of a template:

```bash
mosaicx template history EchoReport
```

**Example output:**

```
EchoReport History

  Version  Date
  v1       2026-02-10 14:23
  v2       2026-02-11 09:15
  current  2026-02-17 10:30

  2 archived version(s) + current
```

### Compare Versions

See what changed between a previous version and the current version:

```bash
mosaicx template diff EchoReport --version 1
```

**Example output:**

```
EchoReport: v1 vs current

      Section                Detail
  +   pericardial_effusion   (enum)
  +   wall_motion            (str)
  ~   lvef                   type: str -> float

  2 added, 0 removed, 1 modified
```

Legend:
- `+` = added in current version
- `-` = removed from current version
- `~` = modified between versions

### Revert to a Previous Version

If you want to go back to an earlier version:

```bash
mosaicx template revert EchoReport --version 1
```

This archives your current version and restores version 1 as the active template.

## Using Templates for Extraction

### Single Document

Extract structured data from a document using a template:

```bash
# Built-in template
mosaicx extract --document scan.pdf --template chest_ct

# User template (by name)
mosaicx extract --document report.pdf --template EchoReport

# YAML file path
mosaicx extract --document report.pdf --template ./templates/echo.yaml

# With completeness scoring
mosaicx extract --document scan.pdf --template chest_ct --score

# Save output to file
mosaicx extract --document scan.pdf --template chest_ct -o result.json
```

### Batch Processing

Process an entire directory of documents with the same template:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured --template chest_ct
```

All documents in `./reports` will be processed using the `chest_ct` template, and the structured output will be saved to `./structured`.

### Auto Extraction (No Template)

If you run extraction without a template, the LLM infers the structure automatically:

```bash
mosaicx extract --document report.pdf
```

This is useful for exploration, but for consistent, reproducible results, always use a template.

### SDK Usage

Templates work the same way through the Python SDK:

```python
from mosaicx.sdk import extract

# Built-in template
result = extract(text, template="chest_ct")

# YAML file path
result = extract(text, template="./templates/echo.yaml")

# User template
result = extract(text, template="EchoReport")
```

## Migrating from Legacy JSON Schemas

If you have JSON schemas from an earlier version of MOSAICX (stored in `~/.mosaicx/schemas/`), you can convert them all to YAML templates in one step:

```bash
# Preview what will be migrated
mosaicx template migrate --dry-run

# Perform the migration
mosaicx template migrate
```

This converts `~/.mosaicx/schemas/*.json` to `~/.mosaicx/templates/*.yaml`. Existing YAML templates with the same name are not overwritten.

You can also convert individual schemas:

```bash
mosaicx template create --from-json ~/.mosaicx/schemas/EchoReport.json
```

Legacy JSON schemas continue to work via the resolution chain (step 4), so there is no urgency to migrate. But YAML templates are the recommended format going forward.

## Complete Example: Building a Pathology Template

Let's walk through a complete example of building a template for surgical pathology reports.

### Step 1: Generate an Initial Template

Start with a description:

```bash
mosaicx template create \
  --describe "surgical pathology report with specimen site, histologic type, tumor grade, margins, and TNM staging" \
  --name PathologyReport \
  --mode pathology
```

### Step 2: Review the Template

```bash
mosaicx template show PathologyReport
```

Suppose the LLM created sections for specimen_site, histologic_type, tumor_grade, margins, and staging -- all as `str` type.

### Step 3: Refine the Template

The tumor grade should be an enum, and you need additional fields:

```bash
mosaicx template refine PathologyReport \
  --instruction "change tumor_grade to an enum with values well_differentiated, moderately_differentiated, poorly_differentiated, undifferentiated. Add tumor_size_cm as a required float field. Add lymph_nodes_positive and lymph_nodes_examined as optional int fields."
```

### Step 4: Test the Template

Extract from a sample report:

```bash
mosaicx extract --document sample_path_report.pdf --template PathologyReport -o result.json
```

### Step 5: Iterate

If the results need adjustment, refine and re-extract. Version history tracks every iteration:

```bash
mosaicx template history PathologyReport
mosaicx template diff PathologyReport --version 1
```

### Step 6: Share with Your Team

Since templates are plain YAML files, you can commit them to version control:

```bash
cp ~/.mosaicx/templates/PathologyReport.yaml ./project/templates/
git add ./project/templates/PathologyReport.yaml
```

Then anyone on the team can use it:

```bash
mosaicx extract --document report.pdf --template ./project/templates/PathologyReport.yaml
```

## Tips and Best Practices

1. **Start with a built-in template when possible.** If your use case is close to one of the 11 built-in templates, start there and refine rather than building from scratch.

2. **Use enums for categorical data.** Instead of free-text `str`, use `enum` with fixed values for standardized fields like severity, laterality, BI-RADS, Lung-RADS, and similar classifications.

3. **Use lists of objects for repeating structures.** Nodules, lesions, masses, and similar findings that can appear multiple times are best modeled as `list` with an `item` of type `object`.

4. **Write clear descriptions.** The LLM uses field descriptions to understand what to look for. A description like "Maximum diameter in millimeters" is far more useful than "Size".

5. **Mark fields as required judiciously.** Only mark a field as required if it is genuinely always present in the documents you are processing. Optional fields will be extracted when present and left null when absent.

6. **Choose specific types.** Prefer `float` over `str` for measurements, `enum` over `str` for categorical data, and `bool` over `str` for yes/no fields. Specific types produce cleaner, more usable output.

7. **Test on multiple documents.** One document may not reveal all edge cases. Run extraction on a handful of representative documents before batch processing.

8. **Use version control for project templates.** Store templates alongside your code in git. This gives you full history, diffs, and collaboration -- more robust than the built-in `.history/` versioning.

9. **Validate before batch runs.** Run `mosaicx template validate --file template.yaml` before processing thousands of documents to catch syntax errors early.

10. **Use `--score` for quality checks.** The `--score` flag on `extract` computes completeness metrics, showing you which fields were populated and which were missed.

## Troubleshooting

### Template not found

```
Error: Template 'MyTemplate' not found.
```

**Solution:** Check the template name with `mosaicx template list`. Names are case-sensitive. If you are referencing a file path, make sure it ends in `.yaml` or `.yml` and the file exists.

### Template validation failed

```
Error: Template validation failed: Unsupported type 'datetime' in field 'exam_date'
```

**Solution:** Templates only support these types: `str`, `int`, `float`, `bool`, `enum`, `list`, `object`. Use `str` for dates, timestamps, and similar values and parse them downstream if needed.

### Field not extracted

If a field shows up as `null` or missing in extraction output:

- Check that the field description is clear and specific.
- Make sure the field name matches the terminology used in the document.
- Verify the document actually contains that information.
- Try making the field optional if it is not always present.

### Refinement not producing expected results

If `template refine` does not do what you expected:

- Be more specific in your instruction.
- Break complex changes into multiple refinement steps.
- Use `template diff` to see exactly what changed.
- For precise control, edit the YAML file directly in a text editor.

## Schemas (Internal Detail)

Under the hood, MOSAICX uses a JSON `SchemaSpec` format as an intermediate representation. When you create or refine a template via the CLI, the LLM generates a SchemaSpec internally, which is then converted to YAML. When you use a template for extraction, the YAML is compiled into a Pydantic model class at runtime.

You do not need to interact with SchemaSpec directly. It exists for backward compatibility with older `~/.mosaicx/schemas/*.json` files and as a bridge between the LLM generation pipeline and the YAML template format. If you have legacy JSON schemas, use `mosaicx template migrate` to convert them.

## CLI Command Reference

| Command | Description |
|---------|-------------|
| `mosaicx template list` | List all available templates (built-in and user) |
| `mosaicx template show <name>` | Display template structure and metadata |
| `mosaicx template create --describe "..."` | Generate a template from a description |
| `mosaicx template create --from-document report.pdf` | Generate a template from a sample document |
| `mosaicx template create --from-url <url>` | Generate a template from web page content |
| `mosaicx template create --from-radreport <id>` | Generate a template from a RadReport ID |
| `mosaicx template create --from-json schema.json` | Convert a JSON schema to YAML |
| `mosaicx template validate --file template.yaml` | Validate a template file |
| `mosaicx template refine <name> --instruction "..."` | Refine a template with LLM instructions |
| `mosaicx template migrate` | Convert all legacy JSON schemas to YAML |
| `mosaicx template history <name>` | Show version history of a user template |
| `mosaicx template diff <name> --version N` | Compare current template to version N |
| `mosaicx template revert <name> --version N` | Revert a template to version N |

## Next Steps

1. Browse the built-in templates with `mosaicx template list` and `mosaicx template show`.
2. Create a template for your document type.
3. Test it on a few sample documents with `mosaicx extract --template ... --document ...`.
4. Refine based on results.
5. Use it for batch processing with `mosaicx batch --template ...`.

For more information:
- `mosaicx extract --help` for extraction options
- `mosaicx batch --help` for batch processing options
- `mosaicx template --help` for all template commands

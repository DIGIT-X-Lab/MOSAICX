# Schemas and Templates

This guide explains how to create, manage, and use schemas and YAML templates in MOSAICX. Whether you're extracting patient vitals from a clinical note or structured findings from a radiology report, schemas and templates give you precise control over what data gets extracted and how it's structured.

## What are Schemas?

A schema defines WHAT you want to extract from a document. Instead of letting the LLM decide what to extract, you tell it exactly which fields to look for and what types they should be.

Think of a schema as a form template. When you read a document, you want to fill out specific fields: patient name (text), age (number), diagnosis (text), medications (list of medications). The schema tells the LLM exactly what information to find and how to structure it.

**Example:** If you're extracting echocardiography reports, you might want:
- `patient_name` (text)
- `lvef` (number) - left ventricular ejection fraction
- `valve_grades` (list of text) - assessment of each valve
- `impression` (text) - the radiologist's conclusion

Schemas are saved as JSON files in `~/.mosaicx/schemas/` and can be reused across many documents.

## Creating a Schema

There are three ways to create a schema: from a description, from a document, or with example text.

### From a Description

The simplest way is to describe what you want in plain English:

```bash
mosaicx schema generate --description "echocardiography report with LVEF, valve grades, and clinical impression"
```

The LLM will create a schema with appropriate field names, types, and descriptions. It automatically chooses the best types for your fields.

**Optionally name your schema:**

```bash
mosaicx schema generate \
  --description "echocardiography report with LVEF, valve grades, and clinical impression" \
  --name EchoReport
```

If you don't provide a name, the LLM will choose one for you (usually something like "EchocardiographyReport").

### From a Document

If you have a sample document, the LLM can read it and infer what fields to extract:

```bash
mosaicx schema generate --from-document sample_report.pdf
```

This is useful when you have a complex document and want the LLM to figure out the structure for you.

### With Example Text

You can also provide example text to help the LLM understand what you're looking for:

```bash
mosaicx schema generate \
  --description "patient vitals from nursing notes" \
  --example-text "BP: 120/80, HR: 72, Temp: 98.6F, SpO2: 98%, Pain: 3/10"
```

The example text helps ground the LLM's understanding of the format and terminology used in your documents.

### Save to a Custom Location

By default, schemas are saved to `~/.mosaicx/schemas/`. You can save to a different location:

```bash
mosaicx schema generate \
  --description "vital signs" \
  --output ./my_schemas/vitals.json
```

## Managing Schemas

### List All Schemas

See all your saved schemas:

```bash
mosaicx schema list
```

**Example output:**

```
Saved Schemas

Name                Fields  Description
EchoReport          7       Echocardiography report with cardiac measurements
PathologyReport     12      Surgical pathology report
VitalSigns          6       Patient vital signs from nursing documentation

3 schema(s) saved in /Users/yourname/.mosaicx/schemas
```

### View a Schema

See the detailed structure of a schema:

```bash
mosaicx schema show EchoReport
```

**Example output:**

```
EchoReport

Echocardiography report with cardiac measurements

Field              Type           Req  Description
patient_name       str            ✓    Patient's full name
exam_date          str            ✓    Date of echocardiogram
lvef               float          ✓    Left ventricular ejection fraction (percentage)
valve_grades       list[str]      ✓    Assessment grades for each cardiac valve
wall_motion        enum(...)      —    Wall motion abnormality severity
pericardial_eff    enum(...)      —    Pericardial effusion severity
impression         str            ✓    Clinical impression and recommendations

7 fields
```

The **Req** column shows whether a field is required (✓) or optional (—).

## Refining a Schema

After creating a schema, you can modify it without starting from scratch. There are two approaches: command-line flags for simple changes, or LLM-driven refinement for complex changes.

### Add a Field

Add a new required field:

```bash
mosaicx schema refine --schema EchoReport --add "rvsp: float"
```

Add an optional field with a description:

```bash
mosaicx schema refine --schema EchoReport \
  --add "hospital: str" \
  --optional \
  --description "Hospital name where exam was performed"
```

### Remove a Field

Remove a field you no longer need:

```bash
mosaicx schema refine --schema EchoReport --remove clinical_impression
```

### Rename a Field

Change a field's name (the type and other attributes stay the same):

```bash
mosaicx schema refine --schema EchoReport --rename "lvef=lvef_percent"
```

The format is `old_name=new_name`.

### LLM-Driven Refinement

For more complex changes, describe what you want in natural language:

```bash
mosaicx schema refine --schema EchoReport \
  --instruction "add a field for pericardial effusion severity as an enum with values none, trivial, small, moderate, large"
```

The LLM will update the schema according to your instruction. This is useful when you want to:
- Change field types
- Add multiple fields at once
- Reorganize the schema structure
- Add complex nested fields

**Example output:**

```
Schema refined — evolution, not revolution
+ pericardial_effusion (enum)
Model: EchoReport
Fields: patient_name, exam_date, lvef, valve_grades, wall_motion, pericardial_effusion, impression
```

## Version History

Every time you generate or refine a schema, MOSAICX automatically archives the previous version. This means you never lose your work and can always roll back if needed.

### View History

See all archived versions of a schema:

```bash
mosaicx schema history EchoReport
```

**Example output:**

```
EchoReport History

Version  Fields  Date
v1       5       2026-02-10 14:23
v2       6       2026-02-11 09:15
v3       7       2026-02-12 16:42
current  7       2026-02-17 10:30

3 archived version(s) + current
```

### Compare Versions

See what changed between a previous version and the current version:

```bash
mosaicx schema diff EchoReport --version 1
```

**Example output:**

```
EchoReport: v1 vs current

    Field                 Detail
+   valve_grades          (list[str])
+   pericardial_effusion  (enum)
~   lvef                  type: int -> float

1 added, 0 removed, 1 modified
```

Legend:
- `+` = added in current version
- `-` = removed from current version
- `~` = modified between versions

### Revert to a Previous Version

If you made a mistake or want to go back to an earlier version:

```bash
mosaicx schema revert EchoReport --version 2
```

This archives your current version and restores version 2 as the active schema.

**Example output:**

```
Reverted EchoReport to v2 (archived current as v4)
- pericardial_effusion
~ lvef (changed)
```

## Using Schemas for Extraction

Once you have a schema, use it to extract structured data from documents.

### Single Document Extraction

Extract from one document:

```bash
mosaicx extract --document report.pdf --schema EchoReport
```

The output will be structured exactly according to your schema.

### Batch Extraction

Process multiple documents with the same schema:

```bash
mosaicx batch --input-dir ./reports --output-dir ./structured --schema EchoReport
```

All documents in `./reports` will be processed using the `EchoReport` schema, and the structured output will be saved to `./structured`.

## Schema JSON Format

Under the hood, schemas are stored as JSON files. Here's what they look like:

```json
{
  "class_name": "EchoReport",
  "description": "Echocardiography report with cardiac measurements",
  "fields": [
    {
      "name": "patient_name",
      "type": "str",
      "description": "Patient's full name",
      "required": true,
      "enum_values": null
    },
    {
      "name": "lvef",
      "type": "float",
      "description": "Left ventricular ejection fraction (percentage)",
      "required": true,
      "enum_values": null
    },
    {
      "name": "valve_grades",
      "type": "list[str]",
      "description": "Assessment grades for each cardiac valve",
      "required": true,
      "enum_values": null
    },
    {
      "name": "severity",
      "type": "enum",
      "description": "Overall severity assessment",
      "required": false,
      "enum_values": ["normal", "mild", "moderate", "severe"]
    }
  ]
}
```

### Supported Field Types

- `str` - Text string
- `int` - Integer number
- `float` - Decimal number
- `bool` - True/false value
- `list[X]` - List of items (where X is any type above, e.g., `list[str]`, `list[int]`)
- `enum` - One of a fixed set of values (requires `enum_values`)

Each field has:
- `name` - The field identifier (must be a valid Python variable name)
- `type` - The data type
- `description` - Human-readable explanation
- `required` - Whether the field must be present in extracted data
- `enum_values` - For enum types, the list of allowed values

You can manually edit these JSON files if needed, but it's usually easier to use the CLI commands.

## YAML Templates

Templates are an alternative to schemas. While schemas are JSON files dynamically generated by the LLM, templates are static YAML files you write yourself. Templates are useful when:

- You want to version-control your extraction structure with your project
- You need to share extraction definitions with colleagues
- You want fine-grained control over nested structures
- You're defining a standard format for your organization

### YAML Template Format

Templates use a YAML format with sections and fields:

```yaml
name: CTChestTemplate
description: "Structured CT chest report"
sections:
  - name: exam_info
    type: object
    required: true
    description: "Basic exam information"
    fields:
      - name: exam_type
        type: str
        required: true
        description: "Type of imaging exam"
      - name: exam_date
        type: str
        required: true
        description: "Date exam was performed"
      - name: protocol
        type: str
        required: false
        description: "Imaging protocol used"

  - name: findings
    type: list
    required: true
    description: "List of imaging findings"
    item:
      type: object
      fields:
        - name: anatomy
          type: str
          required: true
          description: "Anatomical location"
        - name: observation
          type: str
          required: true
          description: "What was observed"
        - name: severity
          type: enum
          required: false
          description: "Severity assessment"
          values: ["mild", "moderate", "severe"]
        - name: measurements
          type: list
          required: false
          description: "Quantitative measurements"
          item:
            type: object
            fields:
              - name: dimension
                type: str
                required: true
              - name: value
                type: float
                required: true
              - name: unit
                type: str
                required: true

  - name: impression
    type: str
    required: true
    description: "Radiologist's clinical impression"
```

This template demonstrates:
- **Simple fields** - `exam_type`, `exam_date`, `impression`
- **Nested objects** - `exam_info` contains multiple fields
- **Lists of objects** - `findings` is a list where each item has `anatomy`, `observation`, etc.
- **Deeply nested structures** - `findings.measurements` is a list of measurement objects
- **Enums** - `severity` must be one of the specified values

### Using Templates

Extract from a document using a template:

```bash
mosaicx extract --document report.pdf --template ./templates/ct_chest.yaml
```

### Validating Templates

Check if your YAML template is valid before using it:

```bash
mosaicx template validate --file ./templates/ct_chest.yaml
```

**Example output:**

```
Template is valid — you shall pass
Model: CTChestTemplate
Fields: exam_info, findings, impression
```

If there are errors (invalid field types, missing required attributes, etc.), you'll get a detailed error message.

### Built-in Templates

MOSAICX includes built-in templates for common radiology reports:

```bash
mosaicx template list
```

**Example output:**

```
Templates

Name           Exam Type      RadReport ID  Description
generic        generic        —             Generic radiology report
chest_ct       chest_ct       RDES3         Chest CT report
chest_xr       chest_xr       RDES2         Chest X-ray report
brain_mri      brain_mri      RDES28        Brain MRI report
abdomen_ct     abdomen_ct     RDES44        Abdomen CT report
mammography    mammography    RDES4         Mammography report
thyroid_us     thyroid_us     RDES72        Thyroid ultrasound report
lung_ct        lung_ct        RDES195       Lung CT screening report
msk_mri        msk_mri        —             MSK MRI report
cardiac_mri    cardiac_mri    RDES214       Cardiac MRI report
pet_ct         pet_ct         RDES76        PET/CT report

11 template(s) registered
```

These templates are based on RadReport standards and can be used directly or as examples for creating your own templates.

## Schema vs Template: When to Use Which

Both schemas and templates define extraction structure, but they serve different purposes:

| Aspect | Schema | Template |
|--------|--------|----------|
| **Format** | JSON | YAML |
| **Creation** | LLM-generated from descriptions or documents | Manually written |
| **Storage** | `~/.mosaicx/schemas/` (user directory) | Your project directory |
| **Modification** | CLI commands (`schema refine`) or LLM instructions | Edit YAML file directly |
| **Version Control** | Automatic history in `.history/` subdirectory | Standard git/version control |
| **Sharing** | Copy JSON files | Commit YAML to repository |
| **Best For** | Quick iteration, exploratory work, personal projects | Team projects, production systems, standards compliance |
| **Dynamic Updates** | Easy - just refine with instructions | Manual - edit YAML |
| **Nested Structures** | Supported (via LLM instructions) | Fully supported with clear syntax |
| **Learning Curve** | Low - natural language descriptions | Medium - YAML syntax and structure |

### Use Schemas When:

- You're exploring what to extract from new document types
- You want the LLM to help design the structure
- You're working on personal or exploratory projects
- You want to quickly iterate on the structure
- You need simple, flat structures

### Use Templates When:

- You need complex nested data structures
- You're working in a team and need version control
- You want to define organization-wide standards
- You need precise control over every field attribute
- You're building production data pipelines
- You want to share extraction definitions with non-MOSAICX users

### Can I Convert Between Them?

Yes! You can manually convert in either direction:

**Schema to Template:** Export the schema JSON and rewrite it as YAML with the template format.

**Template to Schema:** Save the template structure as a schema JSON file in `~/.mosaicx/schemas/`.

In practice, start with schemas for exploration, then convert to templates when you're ready to standardize.

## Complete Example: Building a Pathology Schema

Let's walk through a complete example of building a schema for surgical pathology reports.

### Step 1: Generate Initial Schema

Start with a description:

```bash
mosaicx schema generate \
  --description "surgical pathology report with specimen site, histologic type, tumor grade, margins, and staging" \
  --name PathologyReport
```

### Step 2: Review the Schema

```bash
mosaicx schema show PathologyReport
```

Suppose the LLM created these fields:
- specimen_site (str)
- histologic_type (str)
- tumor_grade (str)
- margins (str)
- staging (str)

### Step 3: Refine the Schema

The tumor grade should be an enum, not free text:

```bash
mosaicx schema refine --schema PathologyReport \
  --instruction "change tumor_grade to an enum with values well_differentiated, moderately_differentiated, poorly_differentiated, undifferentiated"
```

Add a field for tumor size:

```bash
mosaicx schema refine --schema PathologyReport \
  --add "tumor_size_cm: float" \
  --description "Maximum tumor dimension in centimeters"
```

Add optional fields:

```bash
mosaicx schema refine --schema PathologyReport \
  --add "lymph_nodes_positive: int" \
  --optional \
  --description "Number of lymph nodes with metastatic disease"

mosaicx schema refine --schema PathologyReport \
  --add "lymph_nodes_examined: int" \
  --optional \
  --description "Total number of lymph nodes examined"
```

### Step 4: Test the Schema

Extract from a sample report:

```bash
mosaicx extract --document sample_path_report.pdf --schema PathologyReport --output result.json
```

### Step 5: Iterate

If you need changes, refine the schema and re-run extraction. The version history keeps track of all your iterations:

```bash
mosaicx schema history PathologyReport
```

## Tips and Best Practices

### For Schemas:

1. **Start broad, then refine** - Generate an initial schema and refine it based on real documents
2. **Use enums for categorical data** - Instead of free text, use enums for standardized values (severity, laterality, modality)
3. **Keep field names consistent** - Use snake_case and be consistent across schemas
4. **Add good descriptions** - The LLM uses field descriptions to understand what to extract
5. **Test on multiple documents** - One document might not reveal all edge cases

### For Templates:

1. **Use nested objects** - Group related fields together in objects
2. **Make lists explicit** - Use `item` to define what each list element contains
3. **Document with descriptions** - Every field should have a clear description
4. **Validate early** - Run `template validate` before using in production
5. **Follow naming conventions** - Match your organization's data standards

### General:

1. **Required vs optional** - Mark fields as required only if they're always present
2. **Type carefully** - Choose the most specific type (enum > bool > str)
3. **Version control** - Schemas have automatic versioning, templates need git
4. **Test extraction** - Always test on real documents before batch processing
5. **Iterate** - Extraction is rarely perfect on the first try

## Troubleshooting

### Schema not found

```
Error: Schema 'MySchema' not found in /Users/yourname/.mosaicx/schemas
```

**Solution:** Check the schema name with `mosaicx schema list`. Schema names are case-sensitive.

### Template validation failed

```
Error: Template validation failed: Unsupported type 'datetime' in field 'exam_date'
```

**Solution:** Templates only support basic types: str, int, float, bool, enum, list, object. Use `str` for dates and parse them later if needed.

### Field not extracted

If a field shows up as `null` or missing in extraction:
- Check the field description is clear
- Make sure the field name matches the terminology in the document
- Verify the document actually contains that information
- Try making the field optional if it's not always present

### Refinement not working

If `schema refine --instruction` doesn't work as expected:
- Be more specific in your instruction
- Try breaking complex changes into multiple refinement steps
- Check `schema diff` to see what actually changed
- Use CLI flags (`--add`, `--remove`, `--rename`) for simple changes instead

## Next Steps

Now that you understand schemas and templates:

1. Create a schema for your document type
2. Test it on a few sample documents
3. Refine based on results
4. Use it for batch processing
5. Convert to a template if you need to share or version control it

For more information:
- See `mosaicx extract --help` for extraction options
- See `mosaicx batch --help` for batch processing
- See the getting started guide for a complete workflow example

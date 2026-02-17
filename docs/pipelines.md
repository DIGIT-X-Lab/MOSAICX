# Pipelines

MOSAICX provides 6 specialized pipelines for processing medical documents. Each pipeline is a DSPy module — a chain of LLM calls designed for a specific document type or task. You can use these pipelines via CLI commands or optimize them with labeled training data using DSPy's GEPA (Generalized Few-Shot Prompting Algorithm) or MIPROv2 optimizers.

This guide documents what each pipeline does, what inputs it expects, what outputs it produces, and how to create JSONL training records for optimization and evaluation.

---

## Radiology Pipeline (`radiology`)

The radiology pipeline converts free-text radiology reports into structured data with standardized sections, findings, measurements, and actionable impressions.

### CLI Command

```bash
mosaicx extract --document report.pdf --mode radiology
```

### Module

`RadiologyReportStructurer` from `mosaicx.pipelines.radiology`

### What It Does

A 5-step chain that processes radiology reports:

1. **ClassifyExamType** — Identifies the exam modality and body region from the report header (e.g., "CT Chest", "MRI Brain").
2. **ParseReportSections** — Splits the raw report text into standard sections: indication, comparison, technique, findings, and impression.
3. **ExtractTechnique** — Parses the technique section to extract modality, body region, contrast details, and protocol information.
4. **ExtractRadFindings** — Extracts structured findings from the findings section, including anatomy, observations, measurements, severity, and changes from prior studies.
5. **ExtractImpression** — Extracts actionable impression items from the impression section, including clinical categories (e.g., Lung-RADS, BI-RADS), ICD-10 codes, and actionability flags.

Each step is independently optimizable using DSPy's compilation methods.

### Inputs

- `report_text` (str, required) — Full text of the radiology report
- `report_header` (str, optional) — Report header or title. If empty, the first 200 characters of `report_text` are used

### Outputs

- `exam_type` (str) — Classified exam type (e.g., "CT Chest with Contrast")
- `sections` (ReportSections) — Parsed report sections:
  - `indication` (str) — Clinical indication
  - `comparison` (str) — Prior comparison studies
  - `technique` (str) — Imaging technique description
  - `findings` (str) — Raw findings text
  - `impression` (str) — Raw impression text
- `modality` (str) — Imaging modality (CT, MRI, US, XR, etc.)
- `body_region` (str) — Body region imaged
- `contrast` (str) — Contrast agent details or "none"
- `protocol` (str) — Protocol name or description
- `findings` (list of RadReportFinding) — Each finding contains:
  - `anatomy` (str) — Anatomical location
  - `radlex_id` (str, optional) — RadLex ontology identifier
  - `observation` (str) — Short label (e.g., "nodule", "mass")
  - `description` (str) — Full description
  - `measurement` (Measurement, optional) — Quantitative measurement with value, unit, dimension, and prior value
  - `change_from_prior` (ChangeType, optional) — Change status (new/stable/increased/decreased/resolved), prior date, prior measurement
  - `severity` (str, optional) — Severity grade
  - `template_field_id` (str, optional) — RadReport template field ID
- `impressions` (list of ImpressionItem) — Each impression contains:
  - `statement` (str) — Impression text
  - `category` (str, optional) — Classification (e.g., "Lung-RADS 3", "BI-RADS 4")
  - `icd10_code` (str, optional) — Associated ICD-10 code
  - `actionable` (bool) — Whether follow-up is required
  - `finding_refs` (list of int) — Indices of related findings

### Example JSONL Record

For training or evaluation, create a JSONL file where each line is a JSON object containing the inputs and expected outputs:

```json
{"report_text": "CT CHEST WITH CONTRAST\n\nINDICATION: 65-year-old male with history of smoking, evaluate for lung nodules.\n\nCOMPARISON: CT chest from 2023-01-15.\n\nTECHNIQUE: Volumetric axial CT images of the chest were obtained following intravenous administration of 100 mL Omnipaque 350. Images reconstructed at 1.25 mm intervals.\n\nFINDINGS: A 5 mm solid nodule is present in the right upper lobe (series 3, image 45). This measures 3 mm on the prior study, representing interval growth. No mediastinal or hilar lymphadenopathy. Heart size is normal. No pleural effusion.\n\nIMPRESSION:\n1. Growing 5 mm right upper lobe nodule, increased from 3 mm. Recommend 3-month follow-up CT per Lung-RADS 4A.\n2. Otherwise unremarkable chest CT.", "report_header": "CT CHEST WITH CONTRAST", "exam_type": "CT Chest with Contrast", "findings": [{"anatomy": "right upper lobe", "observation": "nodule", "description": "5 mm solid nodule", "measurement": {"value": 5, "unit": "mm", "dimension": "diameter", "prior_value": 3}, "change_from_prior": {"status": "increased", "prior_date": "2023-01-15", "prior_measurement": {"value": 3, "unit": "mm", "dimension": "diameter"}}, "severity": null}], "impressions": [{"statement": "Growing 5 mm right upper lobe nodule, increased from 3 mm. Recommend 3-month follow-up CT per Lung-RADS 4A.", "category": "Lung-RADS 4A", "icd10_code": null, "actionable": true}]}
```

---

## Pathology Pipeline (`pathology`)

The pathology pipeline converts free-text surgical pathology and biopsy reports into structured data with specimen details, microscopic findings, diagnoses, staging, and biomarker results.

### CLI Command

```bash
mosaicx extract --document report.pdf --mode pathology
```

### Module

`PathologyReportStructurer` from `mosaicx.pipelines.pathology`

### What It Does

A 5-step chain that processes pathology reports:

1. **ClassifySpecimenType** — Identifies the specimen or procedure type from the report header (e.g., "Biopsy - Prostate", "Resection - Colon").
2. **ParsePathSections** — Splits the raw report into standard sections: clinical history, gross description, microscopic description, diagnosis, and ancillary studies.
3. **ExtractSpecimenDetails** — Parses the gross description to extract anatomical site, laterality, procedure type, specimen dimensions, and number of specimens received.
4. **ExtractMicroscopicFindings** — Extracts structured findings from the microscopic section, including histologic type, grade, margin status, and invasion patterns.
5. **ExtractPathDiagnosis** — Extracts final diagnoses with WHO classification, TNM staging, ICD-O codes, biomarker results (IHC, molecular), and ancillary study summaries.

### Inputs

- `report_text` (str, required) — Full text of the pathology report
- `report_header` (str, optional) — Report header or title. If empty, the first 200 characters of `report_text` are used

### Outputs

- `specimen_type` (str) — Classified specimen type
- `sections` (PathSections) — Parsed report sections:
  - `clinical_history` (str) — Clinical history
  - `gross_description` (str) — Gross/macroscopic description
  - `microscopic` (str) — Microscopic description
  - `diagnosis` (str) — Final diagnosis section text
  - `ancillary_studies` (str) — Ancillary studies (IHC, molecular, flow)
  - `comment` (str) — Additional comments
- `site` (str) — Anatomical site of the specimen
- `laterality` (str) — Laterality (left/right/bilateral/N/A)
- `procedure` (str) — Procedure type (biopsy, excision, resection)
- `dimensions` (str) — Specimen dimensions (e.g., "3.2 x 2.1 x 1.5 cm")
- `specimens_received` (int) — Number of specimens or parts received
- `findings` (list of PathFinding) — Each finding contains:
  - `description` (str) — Finding description
  - `histologic_type` (str, optional) — Histologic type (e.g., "adenocarcinoma", "squamous cell")
  - `grade` (str, optional) — Histologic grade (e.g., "G2", "Gleason 3+4=7", "Nottingham 2")
  - `margins` (str, optional) — Margin status (positive/negative/close + distance)
  - `lymphovascular_invasion` (str, optional) — LVI status (present/absent/indeterminate)
  - `perineural_invasion` (str, optional) — PNI status (present/absent/indeterminate)
- `diagnoses` (list of PathDiagnosis) — Each diagnosis contains:
  - `diagnosis` (str) — Primary diagnosis text
  - `who_classification` (str, optional) — WHO tumor classification
  - `tnm_stage` (str, optional) — Pathologic TNM stage (e.g., "pT2 pN1a")
  - `icd_o_morphology` (str, optional) — ICD-O morphology code (e.g., "8500/3")
  - `biomarkers` (list of Biomarker) — Each biomarker has:
    - `name` (str) — Biomarker name (e.g., "ER", "PR", "HER2", "Ki-67")
    - `result` (str) — Result (e.g., "positive (95%)", "negative", "3+")
    - `method` (str, optional) — Method (e.g., "IHC", "FISH", "PCR")
  - `ancillary_results` (str, optional) — Summary of ancillary studies

### Example JSONL Record

```json
{"report_text": "SURGICAL PATHOLOGY REPORT\n\nSPECIMEN: A. Left breast, core biopsy.\n\nCLINICAL HISTORY: 52-year-old female with palpable left breast mass. BI-RADS 5 lesion on mammography.\n\nGROSS DESCRIPTION: Received in formalin labeled 'left breast 9 o'clock' are three cores of tan-pink tissue measuring 1.5 cm in aggregate length and 0.1 cm in diameter. Entirely submitted in cassette A1.\n\nMICROSCOPIC DESCRIPTION: Sections show invasive ductal carcinoma, grade 2 (tubule formation 3, nuclear pleomorphism 2, mitotic count 1). No lymphovascular invasion is identified. Adjacent areas show ductal carcinoma in situ, solid and cribriform patterns, intermediate nuclear grade.\n\nIMMUNOHISTOCHEMISTRY:\nER: Positive (95% of tumor cells, strong intensity)\nPR: Positive (80% of tumor cells, moderate intensity)\nHER2: Negative (1+ by IHC)\nKi-67: 18%\n\nDIAGNOSIS:\nA. Left breast, 9 o'clock, core biopsy:\n   - Invasive ductal carcinoma, Nottingham grade 2 (3+2+1=6/9)\n   - Ductal carcinoma in situ, solid and cribriform, intermediate grade\n   - Immunohistochemistry: ER+, PR+, HER2-\n   - Molecular subtype: Luminal A", "report_header": "SURGICAL PATHOLOGY REPORT", "specimen_type": "Biopsy - Breast", "site": "left breast, 9 o'clock", "laterality": "left", "procedure": "biopsy", "dimensions": "1.5 cm aggregate length, 0.1 cm diameter", "specimens_received": 3, "findings": [{"description": "Invasive ductal carcinoma with adjacent DCIS", "histologic_type": "invasive ductal carcinoma", "grade": "Nottingham grade 2 (3+2+1=6/9)", "margins": null, "lymphovascular_invasion": "absent", "perineural_invasion": null}], "diagnoses": [{"diagnosis": "Invasive ductal carcinoma, Nottingham grade 2", "who_classification": null, "tnm_stage": null, "icd_o_morphology": "8500/3", "biomarkers": [{"name": "ER", "result": "positive (95%, strong)", "method": "IHC"}, {"name": "PR", "result": "positive (80%, moderate)", "method": "IHC"}, {"name": "HER2", "result": "negative (1+)", "method": "IHC"}, {"name": "Ki-67", "result": "18%", "method": "IHC"}], "ancillary_results": "Molecular subtype: Luminal A"}]}
```

---

## Extract Pipeline (`extract`)

The extract pipeline provides flexible document extraction in two modes: Auto mode (infers schema from document) and Schema mode (extracts into a provided Pydantic model).

### CLI Commands

```bash
# Auto mode: infer schema from document
mosaicx extract --document report.pdf

# Schema mode: extract using a saved schema
mosaicx extract --document report.pdf --schema PatientIntakeForm
```

### Module

`DocumentExtractor` from `mosaicx.pipelines.extraction`

### What It Does

**Auto mode** (no `output_schema` provided):
1. **InferSchemaFromDocument** — LLM analyzes the document and generates a SchemaSpec describing what fields should be extracted.
2. Schema compilation — The SchemaSpec is compiled into a Pydantic model using `pydantic.create_model()`.
3. **Extract** — Single-step extraction into the inferred model.

**Schema mode** (`output_schema` provided):
1. **Extract** — Single-step ChainOfThought extraction directly into the provided Pydantic model.

### Inputs

- `document_text` (str, required) — Full text of the document to extract from

### Outputs

**Auto mode:**
- `extracted` (dict or Pydantic model) — Extracted data matching the inferred schema
- `inferred_schema` (SchemaSpec) — The schema specification generated by the LLM

**Schema mode:**
- `extracted` (dict or Pydantic model) — Extracted data matching the provided schema

### Example JSONL Record (Auto Mode)

```json
{"document_text": "PATIENT INTAKE FORM\n\nName: John Smith\nDate of Birth: 03/15/1975\nMRN: 12345678\n\nChief Complaint: Persistent cough for 3 weeks\n\nAllergies: Penicillin (rash), Sulfa drugs (hives)\n\nCurrent Medications:\n- Lisinopril 10mg daily\n- Metformin 500mg twice daily\n- Atorvastatin 20mg at bedtime\n\nSocial History:\nSmoking: Former smoker, quit 5 years ago, 20 pack-year history\nAlcohol: Occasional, 2-3 drinks per week\n\nFamily History:\nFather: MI at age 62\nMother: Type 2 diabetes, hypertension", "extracted": {"patient_name": "John Smith", "date_of_birth": "03/15/1975", "mrn": "12345678", "chief_complaint": "Persistent cough for 3 weeks", "allergies": ["Penicillin (rash)", "Sulfa drugs (hives)"], "medications": ["Lisinopril 10mg daily", "Metformin 500mg twice daily", "Atorvastatin 20mg at bedtime"], "smoking_status": "Former smoker, quit 5 years ago", "smoking_pack_years": 20, "alcohol_use": "Occasional, 2-3 drinks per week", "family_history": "Father: MI at age 62, Mother: Type 2 diabetes, hypertension"}}
```

### Example JSONL Record (Schema Mode)

For schema mode, you would first define a Pydantic model programmatically or use the schema pipeline. The JSONL record follows the same pattern but the `extracted` field must match your specific schema structure.

---

## Summarize Pipeline (`summarize`)

The summarize pipeline creates a coherent timeline narrative from multiple medical reports for a single patient.

### CLI Command

```bash
mosaicx summarize --dir ./reports --patient P001
```

### Module

`ReportSummarizer` from `mosaicx.pipelines.summarizer`

### What It Does

A 2-step chain that synthesizes multiple reports:

1. **ExtractTimelineEvent** — Extracts a single structured timeline event from each report (parallelizable across reports). Each event captures the date, exam type, key finding, clinical context, and changes from prior studies.
2. **SynthesizeTimeline** — Synthesizes all extracted timeline events into a unified narrative summary that tells the patient's clinical story in chronological order.

### Inputs

- `reports` (list of str, required) — List of report texts to summarize
- `patient_id` (str, optional) — Patient identifier for context

### Outputs

- `events` (list of TimelineEvent) — Each event contains:
  - `date` (str) — Date of the exam or event (ISO 8601)
  - `exam_type` (str) — Type of exam (e.g., "CT chest", "MRI brain")
  - `key_finding` (str) — Primary finding from the report
  - `clinical_context` (str, optional) — Clinical context or indication
  - `change_from_prior` (str, optional) — Change compared to prior exam
- `narrative` (str) — Coherent narrative summary synthesizing all events

### Example JSONL Record

```json
{"reports": ["CT CHEST 2024-01-15\n\nINDICATION: Follow-up of lung nodule.\n\nCOMPARISON: CT chest from 2023-07-10.\n\nFINDINGS: The 4 mm right upper lobe nodule is unchanged in size compared to prior. No new nodules. No lymphadenopathy.\n\nIMPRESSION: Stable 4 mm RUL nodule. Continue annual surveillance.", "CT CHEST 2024-07-18\n\nINDICATION: Annual surveillance of lung nodule.\n\nCOMPARISON: CT chest from 2024-01-15.\n\nFINDINGS: The previously noted 4 mm right upper lobe nodule now measures 7 mm, representing interval growth. No mediastinal adenopathy.\n\nIMPRESSION: Growing RUL nodule, increased from 4 mm to 7 mm over 6 months. Recommend 3-month follow-up or biopsy per Lung-RADS 4A.", "BRONCHOSCOPY WITH BIOPSY 2024-08-02\n\nPROCEDURE: Bronchoscopy with transbronchial biopsy of right upper lobe mass.\n\nFINDINGS: Endobronchial lesion identified in the right upper lobe. Biopsies obtained.\n\nPATHOLOGY: Adenocarcinoma, favor primary lung origin. TTF-1 positive, PD-L1 80%."], "patient_id": "P001", "events": [{"date": "2024-01-15", "exam_type": "CT chest", "key_finding": "Stable 4 mm right upper lobe nodule", "clinical_context": "Follow-up of lung nodule", "change_from_prior": "Unchanged from prior"}, {"date": "2024-07-18", "exam_type": "CT chest", "key_finding": "Right upper lobe nodule grown to 7 mm", "clinical_context": "Annual surveillance", "change_from_prior": "Increased from 4 mm to 7 mm over 6 months"}, {"date": "2024-08-02", "exam_type": "Bronchoscopy with biopsy", "key_finding": "Adenocarcinoma of right upper lobe, TTF-1+, PD-L1 80%", "clinical_context": "Biopsy of growing nodule", "change_from_prior": "New diagnosis"}], "narrative": "Patient P001 has been followed for a right upper lobe pulmonary nodule. On CT chest from 2024-01-15, a 4 mm nodule was stable compared to prior imaging from 2023. Surveillance imaging on 2024-07-18 showed interval growth to 7 mm over 6 months, prompting further evaluation. Bronchoscopy with biopsy on 2024-08-02 confirmed adenocarcinoma of the right upper lobe, TTF-1 positive with PD-L1 expression of 80%, consistent with primary lung cancer."}
```

---

## Deidentify Pipeline (`deidentify`)

The deidentify pipeline removes Protected Health Information (PHI) from medical documents using a two-layer approach: LLM-based redaction plus deterministic regex scrubbing.

### CLI Commands

```bash
# Default: remove PHI by replacing with [REDACTED]
mosaicx deidentify --document note.txt

# Pseudonymize: replace PHI with realistic fake values
mosaicx deidentify --document note.txt --mode pseudonymize

# Date shift: shift all dates by a consistent random offset
mosaicx deidentify --document note.txt --mode dateshift

# Regex-only mode (no LLM calls)
mosaicx deidentify --document note.txt --regex-only
```

### Module

`Deidentifier` from `mosaicx.pipelines.deidentifier`

### What It Does

A 2-layer approach to PHI removal:

1. **LLM redaction** (RedactPHI) — A language model identifies and removes context-dependent PHI that cannot be reliably caught by patterns alone, including names, addresses, hospital names, physician names, and dates of birth.
2. **Regex guard** (regex_scrub_phi) — A deterministic regex sweep catches format-based PHI that the LLM might miss, including:
   - Social Security Numbers (SSN): 123-45-6789
   - Phone numbers: (555) 123-4567, 555-123-4567, 555.123.4567
   - Medical Record Numbers (MRN): MRN: 12345678
   - Email addresses: john.doe@hospital.com
   - US date formats: 1/2/2024, 01/02/24

This "belt-and-suspenders" strategy ensures comprehensive PHI removal even if one layer fails.

### Inputs

- `document_text` (str, required) — Full text of the medical document to de-identify
- `mode` (str, optional) — De-identification mode: "remove" (default), "pseudonymize", or "dateshift"

### Outputs

- `redacted_text` (str) — De-identified text with PHI removed or replaced

### PHI Patterns Detected

The regex layer catches the following patterns:
- SSN: `\b\d{3}-\d{2}-\d{4}\b`
- Phone: `\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}`
- MRN: `\bMRN\s*:?\s*\d{6,}\b` (case-insensitive)
- Email: `\b[\w.+-]+@[\w-]+\.[\w.-]+\b`
- US dates: `\b\d{1,2}/\d{1,2}/\d{2,4}\b`

### Example JSONL Record

```json
{"document_text": "CLINIC NOTE\n\nPatient: Jane Smith\nMRN: 87654321\nDOB: 05/23/1968\n\nDate of Visit: 2024-01-15\n\nMs. Smith is a 55-year-old woman who presents for follow-up of hypertension. She reports good adherence to lisinopril 10mg daily. Blood pressure today is 128/82.\n\nContact: (555) 123-4567\nEmail: jane.smith@email.com\nSSN: 123-45-6789\n\nSigned: Dr. Robert Johnson, MD\nRiverside Medical Center", "mode": "remove", "redacted_text": "CLINIC NOTE\n\nPatient: [REDACTED]\nMRN: [REDACTED]\nDOB: [REDACTED]\n\nDate of Visit: [REDACTED]\n\n[REDACTED] is a 55-year-old woman who presents for follow-up of hypertension. She reports good adherence to lisinopril 10mg daily. Blood pressure today is 128/82.\n\nContact: [REDACTED]\nEmail: [REDACTED]\nSSN: [REDACTED]\n\nSigned: [REDACTED]\n[REDACTED]"}
```

---

## Schema Pipeline (`schema`)

The schema pipeline generates Pydantic model schemas from natural language descriptions, enabling you to create custom extraction schemas without writing code.

### CLI Commands

```bash
# Generate schema from description
mosaicx schema generate --description "A patient intake form with demographics, allergies, medications, and social history"

# Generate schema from example document
mosaicx schema generate --from-document example.pdf --name PatientIntake

# Refine an existing schema
mosaicx schema refine PatientIntake "Add a field for emergency contact phone number"

# List all saved schemas
mosaicx schema list

# Show schema details
mosaicx schema show PatientIntake
```

### Module

`SchemaGenerator` from `mosaicx.pipelines.schema_gen`

### What It Does

Generates a SchemaSpec (declarative schema specification) from natural language:

1. **GenerateSchemaSpec** — LLM analyzes the description and optional example text to generate a SchemaSpec JSON object describing the desired Pydantic model structure.
2. Schema compilation — The SchemaSpec is compiled into a concrete Pydantic BaseModel class using `pydantic.create_model()` (no `exec()` or code generation).

The pipeline intelligently chooses field types:
- `enum` with enum_values for categorical fields (modality, severity, laterality)
- `bool` for yes/no or present/absent fields
- `list[str]` for multi-value fields (findings, diagnoses)
- `int` or `float` for numeric measurements
- `str` only for genuinely free-text content (impressions, narratives)

### Inputs

- `description` (str, optional) — Natural language description of the document type to structure
- `example_text` (str, optional) — Optional example document text for grounding
- `document_text` (str, optional) — Optional full document text to infer schema structure from

At least one of `description`, `example_text`, or `document_text` must be provided.

### Outputs

- `schema_spec` (SchemaSpec) — The generated schema specification with:
  - `class_name` (str) — Name for the generated model (PascalCase)
  - `description` (str) — Docstring for the model
  - `fields` (list of FieldSpec) — Each field has:
    - `name` (str) — Field name (snake_case)
    - `type` (str) — Type string (str/int/float/bool/list[X]/enum)
    - `description` (str) — Human-readable description
    - `required` (bool) — Whether the field is required
    - `enum_values` (list of str, optional) — Allowed values for enum types
- `compiled_model` (type[BaseModel]) — The compiled Pydantic model class

### Example JSONL Record

```json
{"description": "A radiology order form with patient demographics, exam type, clinical indication, and ordering physician information", "schema_spec": {"class_name": "RadiologyOrder", "description": "Structured radiology order form", "fields": [{"name": "patient_name", "type": "str", "description": "Patient full name", "required": true}, {"name": "mrn", "type": "str", "description": "Medical record number", "required": true}, {"name": "date_of_birth", "type": "str", "description": "Patient date of birth", "required": true}, {"name": "exam_type", "type": "str", "description": "Type of imaging exam ordered", "required": true}, {"name": "body_region", "type": "str", "description": "Body region to be imaged", "required": true}, {"name": "contrast", "type": "enum", "description": "Contrast administration", "required": true, "enum_values": ["with contrast", "without contrast", "with and without contrast"]}, {"name": "clinical_indication", "type": "str", "description": "Clinical reason for exam", "required": true}, {"name": "stat_order", "type": "bool", "description": "Whether this is a stat/urgent order", "required": false}, {"name": "ordering_physician", "type": "str", "description": "Name of ordering physician", "required": true}, {"name": "callback_number", "type": "str", "description": "Contact number for results", "required": false}]}}
```

---

## Pipeline Summary Table

| Pipeline | CLI Command | Primary Input(s) | Primary Output(s) |
|----------|-------------|------------------|-------------------|
| **Radiology** | `mosaicx extract --mode radiology` | `report_text`, `report_header` | `exam_type`, `sections`, `modality`, `body_region`, `contrast`, `protocol`, `findings`, `impressions` |
| **Pathology** | `mosaicx extract --mode pathology` | `report_text`, `report_header` | `specimen_type`, `sections`, `site`, `laterality`, `procedure`, `dimensions`, `specimens_received`, `findings`, `diagnoses` |
| **Extract** | `mosaicx extract [--schema NAME]` | `document_text` | `extracted`, optionally `inferred_schema` |
| **Summarize** | `mosaicx summarize --dir PATH` | `reports` (list), `patient_id` | `events`, `narrative` |
| **Deidentify** | `mosaicx deidentify [--mode MODE]` | `document_text`, `mode` | `redacted_text` |
| **Schema** | `mosaicx schema generate` | `description`, `example_text`, `document_text` | `schema_spec`, `compiled_model` |

---

## Using JSONL Records for Training and Evaluation

Each pipeline can be optimized using DSPy's compilation methods (GEPA, MIPROv2) with labeled training data. To create a training set:

1. **Create a JSONL file** where each line is a JSON object containing inputs and expected outputs for one example.

2. **Include all required inputs and the ground-truth outputs you want the model to learn.** The optimizer will use these examples to improve prompts and select demonstrations.

3. **For multi-step pipelines** (radiology, pathology), you can optimize individual steps by providing intermediate outputs. For example, for the radiology pipeline, you could provide just `report_text` and `findings` to optimize the ExtractRadFindings step.

4. **Use realistic medical text** in your examples. The quality of your training data directly impacts optimization results.

5. **Load your JSONL file** and pass it to the DSPy optimizer:

```python
import dspy
from mosaicx.pipelines.radiology import RadiologyReportStructurer

# Load training data
with open("radiology_train.jsonl") as f:
    trainset = [dspy.Example(**json.loads(line)).with_inputs("report_text", "report_header")
                for line in f]

# Create and compile the pipeline
pipeline = RadiologyReportStructurer()
optimizer = dspy.GEPA(metric=your_metric_function)
compiled_pipeline = optimizer.compile(pipeline, trainset=trainset)
```

For detailed information on optimization and evaluation, see the DSPy documentation and the MOSAICX optimization guide.

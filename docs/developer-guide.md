# MOSAICX Developer Guide

This guide covers two features for developers who want to extend MOSAICX or use it programmatically:

1. **Creating Custom Pipelines** -- scaffold new DSPy extraction pipelines and wire them into the system.
2. **Python SDK** -- call MOSAICX functions from your own Python code without the CLI.

Prerequisites: You should have MOSAICX installed (`pip install mosaicx`) and be comfortable with Python. Familiarity with DSPy is helpful but not required -- this guide explains everything you need.

---

## Part 1: Creating Custom Pipelines

### Overview

MOSAICX uses [DSPy](https://dspy-docs.vercel.app/) pipelines to extract structured data from clinical documents. Each pipeline is a DSPy Module that chains one or more LLM calls (called "steps") together. For example, the built-in radiology pipeline has 5 steps: classify exam type, parse sections, extract technique, extract findings, and extract impressions.

You can create new pipelines with a single command. The scaffolder generates a working single-step pipeline file, and you then customize it and wire it into the rest of MOSAICX.

### Quick Start

Run the scaffolder from your terminal:

```bash
mosaicx pipeline new cardiology --description "Cardiology report structurer"
```

This creates a new file at `mosaicx/pipelines/cardiology.py` containing a complete, runnable single-step DSPy pipeline. The output will look like:

```
Pipeline scaffolded: /path/to/mosaicx/pipelines/cardiology.py

Next Steps

  The generated pipeline is runnable but needs to be wired
  into the rest of MOSAICX.  Complete these manual steps:

  > Add to mosaicx/pipelines/modes.py _MODE_MODULES dict:
      "cardiology": "mosaicx.pipelines.cardiology",

  > Add to mosaicx/pipelines/modes.py _trigger_lazy_load() _LAZY_CLASS_NAMES dict:
      "cardiology": "CardiologyReportStructurer",

  > Add to mosaicx/evaluation/dataset.py PIPELINE_INPUT_FIELDS dict:
      "cardiology": ["document_text"],

  > Add a metric to mosaicx/evaluation/metrics.py _METRIC_REGISTRY dict:
      "cardiology": <your_metric_function>,

  > Add to mosaicx/evaluation/optimize.py _PIPELINE_REGISTRY dict:
      "cardiology": "mosaicx.pipelines.cardiology.CardiologyReportStructurer",

  > Wire CLI: add `import mosaicx.pipelines.cardiology` alongside other
      pipeline imports in mosaicx/cli.py (extract and batch commands).
```

The name you provide is automatically normalized:

- `cardiology` becomes file `cardiology.py`, class `CardiologyReportStructurer`
- `CardioVascular` becomes file `cardio_vascular.py`, class `CardioVascularReportStructurer`
- `echo-report` becomes file `echo_report.py`, class `EchoReportReportStructurer`

### What Gets Generated

The generated pipeline file has four main sections. Here is what each one does, using `cardiology` as the example.

#### 1. Mode Registration (Eager)

```python
from mosaicx.pipelines.modes import register_mode_info

register_mode_info("cardiology", "Cardiology report structurer")
```

This runs at import time and registers metadata (the name and description) without importing DSPy. This is what allows `mosaicx extract --list-modes` to show your pipeline instantly, even before DSPy loads.

#### 2. DSPy Signature

Inside the `_build_dspy_classes()` function, a DSPy Signature defines the inputs and outputs for your extraction step:

```python
class ExtractCardiology(dspy.Signature):
    """Extract structured information from a cardiology document."""

    document_text: str = dspy.InputField(
        desc="Full text of the cardiology document"
    )
    extracted: str = dspy.OutputField(
        desc="Structured extraction result as JSON"
    )
```

A Signature is a typed contract: it tells DSPy what the LLM receives and what it should produce. The `desc` strings become part of the prompt that DSPy constructs.

#### 3. DSPy Module with forward()

The Module wires Signatures into a pipeline:

```python
class CardiologyReportStructurer(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractCardiology)

    def forward(self, document_text: str) -> dspy.Prediction:
        from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

        metrics = PipelineMetrics()
        tracker = get_tracker()

        with track_step(metrics, "Extract structured data", tracker):
            result = self.extract(document_text=document_text)

        self._last_metrics = metrics

        return dspy.Prediction(
            extracted=result.extracted,
        )
```

Key points:

- `dspy.ChainOfThought` makes the LLM reason step-by-step before answering. Use `dspy.Predict` for simpler steps that do not need reasoning.
- `forward()` is the method DSPy calls when you run the pipeline. It receives the input fields and returns a `dspy.Prediction`.
- The `track_step` context manager records timing and token usage for each step.
- `self._last_metrics` stores metrics so the CLI can display them after extraction.

#### 4. Lazy Loading Boilerplate

```python
_dspy_classes: dict[str, type] | None = None

_DSPY_CLASS_NAMES = frozenset({
    "ExtractCardiology",
    "CardiologyReportStructurer",
})

def __getattr__(name: str):
    global _dspy_classes
    if name in _DSPY_CLASS_NAMES:
        if _dspy_classes is None:
            _dspy_classes = _build_dspy_classes()
        return _dspy_classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

This is the lazy loading pattern. DSPy is a heavy import (~2 seconds). The module-level `__getattr__` ensures DSPy classes are only built when someone actually accesses them (for example, `from mosaicx.pipelines.cardiology import CardiologyReportStructurer`). Until then, importing the module only runs the lightweight `register_mode_info` call.

You do not need to modify this boilerplate. Just make sure `_DSPY_CLASS_NAMES` lists every class defined inside `_build_dspy_classes()`.

### Wiring Checklist

After scaffolding, you need to register your pipeline in 5 places. Below is exactly what to add for a `cardiology` pipeline.

#### a. `mosaicx/pipelines/modes.py` -- Mode Module Mapping

Add your pipeline to the `_MODE_MODULES` dict so that `get_mode("cardiology")` can find and lazy-load it:

```python
_MODE_MODULES: dict[str, str] = {
    "radiology": "mosaicx.pipelines.radiology",
    "pathology": "mosaicx.pipelines.pathology",
    "cardiology": "mosaicx.pipelines.cardiology",       # <-- add this
}
```

Then add the class name to `_LAZY_CLASS_NAMES` inside the `_trigger_lazy_load()` function:

```python
_LAZY_CLASS_NAMES = {
    "radiology": "RadiologyReportStructurer",
    "pathology": "PathologyReportStructurer",
    "cardiology": "CardiologyReportStructurer",          # <-- add this
}
```

#### b. `mosaicx/evaluation/dataset.py` -- Input Fields

Add your pipeline to `PIPELINE_INPUT_FIELDS` so the dataset loader knows which JSONL keys are inputs (everything else is treated as gold-standard labels for scoring):

```python
PIPELINE_INPUT_FIELDS: dict[str, list[str]] = {
    "radiology": ["report_text", "report_header"],
    "pathology": ["report_text", "report_header"],
    "extract": ["document_text"],
    "summarize": ["reports", "patient_id"],
    "deidentify": ["document_text", "mode"],
    "schema": ["description", "example_text", "document_text"],
    "cardiology": ["document_text"],                     # <-- add this
}
```

The field names must match the parameter names of your Module's `forward()` method. If your pipeline takes `document_text`, list `["document_text"]`. If it takes `report_text` and `report_header` like radiology, list both.

#### c. `mosaicx/evaluation/metrics.py` -- Metric Function

Write a scoring function that compares a gold-standard example against a prediction. The function must follow this signature:

```python
def cardiology_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score cardiology extraction quality.

    Components:
    - extracted content quality (weight 0.5)
    - key field presence (weight 0.5)
    """
    from .rewards import extraction_reward

    pred_extracted = _safe_str(getattr(prediction, "extracted", ""))

    # Score based on extraction quality
    if not pred_extracted.strip():
        return 0.0

    # Check that the extraction is non-trivial JSON
    import json
    try:
        parsed = json.loads(pred_extracted)
        if isinstance(parsed, dict) and parsed:
            content_score = 1.0
        else:
            content_score = 0.5
    except (json.JSONDecodeError, TypeError):
        content_score = 0.3

    # Compare against gold if available
    gold_extracted = _safe_str(getattr(example, "extracted", ""))
    if gold_extracted:
        gold_tokens = _token_set(gold_extracted)
        pred_tokens = _token_set(pred_extracted)
        if gold_tokens:
            overlap_score = len(gold_tokens & pred_tokens) / len(gold_tokens)
        else:
            overlap_score = 1.0
    else:
        overlap_score = content_score

    return 0.5 * content_score + 0.5 * overlap_score
```

Then register it in `_METRIC_REGISTRY` at the bottom of the file:

```python
_METRIC_REGISTRY: dict[str, Callable] = {
    "radiology": radiology_metric,
    "pathology": pathology_metric,
    "extract": extraction_metric,
    "summarize": summarizer_metric,
    "deidentify": deidentifier_metric,
    "schema": schema_gen_metric,
    "cardiology": cardiology_metric,                     # <-- add this
}
```

The metric receives two arguments: `example` (a `dspy.Example` with both inputs and gold labels) and `prediction` (the output from your Module's `forward()`). Return a float between 0.0 (worst) and 1.0 (best).

#### d. `mosaicx/evaluation/optimize.py` -- Pipeline Registry

Add your pipeline to `_PIPELINE_REGISTRY` so the optimizer can find and instantiate it:

```python
_PIPELINE_REGISTRY: dict[str, tuple[str, str]] = {
    "radiology":    ("mosaicx.pipelines.radiology",    "RadiologyReportStructurer"),
    "pathology":    ("mosaicx.pipelines.pathology",    "PathologyReportStructurer"),
    "extract":      ("mosaicx.pipelines.extraction",   "DocumentExtractor"),
    "summarize":    ("mosaicx.pipelines.summarizer",   "ReportSummarizer"),
    "deidentify":   ("mosaicx.pipelines.deidentifier", "Deidentifier"),
    "schema":       ("mosaicx.pipelines.schema_gen",   "SchemaGenerator"),
    "cardiology":   ("mosaicx.pipelines.cardiology",   "CardiologyReportStructurer"),  # <-- add this
}
```

The value is a tuple of `(module_path, class_name)`. The optimizer uses these to lazily import and instantiate your pipeline.

#### e. `mosaicx/cli.py` -- CLI Wiring

In the `extract` command and the `batch` command, add an import for your pipeline module alongside the existing imports. Search for lines like:

```python
import mosaicx.pipelines.radiology  # noqa: F401
import mosaicx.pipelines.pathology  # noqa: F401
```

Add your pipeline import next to them:

```python
import mosaicx.pipelines.radiology   # noqa: F401
import mosaicx.pipelines.pathology   # noqa: F401
import mosaicx.pipelines.cardiology  # noqa: F401       # <-- add this
```

There are three places in `cli.py` where these imports appear:

1. Inside the `extract` command's `--list-modes` handler
2. Inside the `extract` command's `--mode` handler
3. Inside the `batch` command's `--mode` handler

Add your import line in all three locations.

### Customizing Your Pipeline

#### Adding More Steps

The scaffolded pipeline has a single step. To add more, define additional Signatures and chain them in `forward()`. Here is how the radiology pipeline chains 5 steps:

```python
def _build_dspy_classes():
    import dspy

    # Step 1
    class ClassifyExamType(dspy.Signature):
        """Classify the type of radiology exam from the report header."""
        report_header: str = dspy.InputField(desc="Header / title portion")
        exam_type: str = dspy.OutputField(desc="Identified exam type")

    # Step 2
    class ParseReportSections(dspy.Signature):
        """Split a radiology report into its standard sections."""
        report_text: str = dspy.InputField(desc="Full text of the report")
        sections: ReportSections = dspy.OutputField(desc="Parsed sections")

    # Wire them together in the Module
    class RadiologyReportStructurer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.classify_exam = dspy.Predict(ClassifyExamType)
            self.parse_sections = dspy.Predict(ParseReportSections)

        def forward(self, report_text, report_header=""):
            # Step 1
            classify_result = self.classify_exam(report_header=report_header)
            exam_type = classify_result.exam_type

            # Step 2: use output of step 1 if needed
            sections_result = self.parse_sections(report_text=report_text)

            return dspy.Prediction(
                exam_type=exam_type,
                sections=sections_result.sections,
            )
```

The pattern is: define a Signature for each step, wire each as `dspy.Predict` or `dspy.ChainOfThought` in `__init__`, then call them sequentially in `forward()`, passing outputs from earlier steps as inputs to later ones.

Remember to:
- Add each new class name to `_DSPY_CLASS_NAMES` in the lazy loading boilerplate.
- Add each new class to the dict returned by `_build_dspy_classes()`.
- Wrap each step call in `track_step()` for metrics.

#### Adding Typed Output Fields (Pydantic Models)

Instead of returning raw strings, you can use Pydantic models as output types. DSPy will parse the LLM output into your model automatically:

```python
from typing import List
from pydantic import BaseModel

class CardiacFinding(BaseModel):
    structure: str       # e.g., "left ventricle", "mitral valve"
    observation: str     # e.g., "mildly dilated"
    severity: str        # e.g., "mild", "moderate", "severe"
    measurement: str     # e.g., "LVEF 45%"

class ExtractCardiacFindings(dspy.Signature):
    """Extract structured cardiac findings from an echo report."""
    findings_text: str = dspy.InputField(desc="Findings section text")
    findings: List[CardiacFinding] = dspy.OutputField(
        desc="List of structured cardiac findings"
    )
```

Place your Pydantic models in `mosaicx/schemas/cardiology/` (or define them inline in the pipeline file for simpler cases). Import them at the top of your pipeline file (outside `_build_dspy_classes()`), since Pydantic does not need DSPy.

#### ChainOfThought vs Predict

- **`dspy.Predict(MySignature)`** -- Direct question-answer. Faster and cheaper. Best for classification, parsing, and simple extraction.
- **`dspy.ChainOfThought(MySignature)`** -- Adds a reasoning step before the answer. Slower but more accurate. Best for complex extraction, finding identification, and impression synthesis.

Rule of thumb: Use `Predict` for steps where the mapping is straightforward (classify exam type, parse sections). Use `ChainOfThought` for steps that require clinical reasoning (extract findings, generate impressions).

#### Adding Metrics Tracking

Every step should be wrapped in `track_step()` for timing and token tracking:

```python
from mosaicx.metrics import PipelineMetrics, get_tracker, track_step

metrics = PipelineMetrics()
tracker = get_tracker()

with track_step(metrics, "Classify exam type", tracker):
    result = self.classify_exam(report_header=header)

with track_step(metrics, "Extract findings", tracker):
    result = self.extract_findings(findings_text=text)

self._last_metrics = metrics
```

The CLI reads `self._last_metrics` after execution to display step-by-step timing and token usage.

### Testing Your Pipeline

#### Create Test Data

Create a JSONL file where each line is a JSON object. Keys that match your `PIPELINE_INPUT_FIELDS` entry are treated as inputs; all other keys are gold-standard labels.

For a cardiology pipeline with `PIPELINE_INPUT_FIELDS["cardiology"] = ["document_text"]`:

```jsonl
{"document_text": "ECHOCARDIOGRAM REPORT\nLV is mildly dilated. LVEF estimated at 40%. Mitral valve shows moderate regurgitation.", "extracted": "{\"lvef\": \"40%\", \"lv_size\": \"mildly dilated\", \"mitral_valve\": \"moderate regurgitation\"}"}
{"document_text": "STRESS TEST REPORT\nPatient exercised for 8 minutes on Bruce protocol. No ST changes. Normal wall motion.", "extracted": "{\"protocol\": \"Bruce\", \"duration\": \"8 minutes\", \"st_changes\": \"none\", \"wall_motion\": \"normal\"}"}
{"document_text": "CARDIAC CATH REPORT\nLAD 70% stenosis. RCA 40% stenosis. LVEF 55% by ventriculography.", "extracted": "{\"lvef\": \"55%\", \"lad_stenosis\": \"70%\", \"rca_stenosis\": \"40%\"}"}
```

Save this as `tests/datasets/cardiology_train.jsonl`.

#### Run Optimization

```bash
mosaicx optimize \
  --pipeline cardiology \
  --trainset tests/datasets/cardiology_train.jsonl \
  --budget light \
  --save ~/.mosaicx/optimized/cardiology_optimized.json
```

Budget presets control the optimization strategy:

| Budget | Strategy | Cost | Time | Min Examples |
|--------|----------|------|------|--------------|
| `light` | BootstrapFewShot | ~$0.50 | ~5 min | 10 |
| `medium` | MIPROv2 | ~$3 | ~20 min | 10 |
| `heavy` | GEPA | ~$10 | ~45 min | 10 |

Start with `light` to verify everything works, then move to `medium` or `heavy` for production quality.

#### Run Evaluation

```bash
mosaicx eval \
  --pipeline cardiology \
  --testset tests/datasets/cardiology_test.jsonl \
  --optimized ~/.mosaicx/optimized/cardiology_optimized.json
```

This runs each test example through your pipeline and scores it with your metric function, then reports mean, median, min, max, and standard deviation.

### Complete Example: Building a Dermatology Pipeline

This walkthrough creates a 2-step dermatology pipeline from scratch.

#### Step 1: Scaffold

```bash
mosaicx pipeline new dermatology --description "Dermatology pathology report structurer"
```

This creates `mosaicx/pipelines/dermatology.py`.

#### Step 2: Add a Second Step

Edit the generated file to add a second Signature and chain them together. Replace the contents of `_build_dspy_classes()`:

```python
def _build_dspy_classes():
    import dspy

    from pydantic import BaseModel
    from typing import List

    class SkinLesion(BaseModel):
        location: str
        morphology: str
        size_mm: str
        margins: str

    # Step 1: Extract lesion details
    class ExtractDermatologyLesions(dspy.Signature):
        """Extract structured lesion descriptions from a dermatology report."""
        document_text: str = dspy.InputField(
            desc="Full text of the dermatology pathology report"
        )
        lesions: List[SkinLesion] = dspy.OutputField(
            desc="List of structured skin lesion descriptions"
        )

    # Step 2: Classify diagnosis
    class ClassifyDermatologyDiagnosis(dspy.Signature):
        """Classify the dermatological diagnosis based on lesion findings."""
        lesions_json: str = dspy.InputField(
            desc="JSON array of extracted lesion descriptions"
        )
        document_text: str = dspy.InputField(
            desc="Original report text for context"
        )
        diagnosis: str = dspy.OutputField(
            desc="Primary dermatological diagnosis"
        )
        malignancy_risk: str = dspy.OutputField(
            desc="Malignancy risk assessment: benign, low, moderate, or high"
        )

    class DermatologyReportStructurer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract_lesions = dspy.ChainOfThought(ExtractDermatologyLesions)
            self.classify_diagnosis = dspy.ChainOfThought(ClassifyDermatologyDiagnosis)

        def forward(self, document_text: str) -> dspy.Prediction:
            from mosaicx.metrics import PipelineMetrics, get_tracker, track_step
            import json

            metrics = PipelineMetrics()
            tracker = get_tracker()

            # Step 1: Extract lesions
            with track_step(metrics, "Extract lesions", tracker):
                lesion_result = self.extract_lesions(document_text=document_text)
            lesions = lesion_result.lesions

            # Step 2: Classify diagnosis using extracted lesions
            lesions_json = json.dumps(
                [l.model_dump() for l in lesions], indent=2
            )
            with track_step(metrics, "Classify diagnosis", tracker):
                diag_result = self.classify_diagnosis(
                    lesions_json=lesions_json,
                    document_text=document_text,
                )

            self._last_metrics = metrics

            return dspy.Prediction(
                lesions=lesions,
                diagnosis=diag_result.diagnosis,
                malignancy_risk=diag_result.malignancy_risk,
            )

    from mosaicx.pipelines.modes import register_mode
    register_mode("dermatology", "Dermatology pathology report structurer")(
        DermatologyReportStructurer
    )

    return {
        "ExtractDermatologyLesions": ExtractDermatologyLesions,
        "ClassifyDermatologyDiagnosis": ClassifyDermatologyDiagnosis,
        "DermatologyReportStructurer": DermatologyReportStructurer,
    }
```

Update `_DSPY_CLASS_NAMES` to list all three classes:

```python
_DSPY_CLASS_NAMES = frozenset({
    "ExtractDermatologyLesions",
    "ClassifyDermatologyDiagnosis",
    "DermatologyReportStructurer",
})
```

#### Step 3: Wire Into Registries

**`mosaicx/pipelines/modes.py`** -- add to `_MODE_MODULES`:

```python
"dermatology": "mosaicx.pipelines.dermatology",
```

And add to `_LAZY_CLASS_NAMES` inside `_trigger_lazy_load()`:

```python
"dermatology": "DermatologyReportStructurer",
```

**`mosaicx/evaluation/dataset.py`** -- add to `PIPELINE_INPUT_FIELDS`:

```python
"dermatology": ["document_text"],
```

**`mosaicx/evaluation/metrics.py`** -- add a metric function:

```python
def dermatology_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Score dermatology extraction quality.

    Components:
    - lesion count similarity (weight 0.3)
    - diagnosis match (weight 0.4)
    - malignancy risk match (weight 0.3)
    """
    # Lesion count similarity
    gold_lesions = _safe_list(getattr(example, "lesions", []))
    pred_lesions = _safe_list(getattr(prediction, "lesions", []))
    gold_count = len(gold_lesions)
    pred_count = len(pred_lesions)
    if gold_count == 0 and pred_count == 0:
        count_score = 1.0
    elif gold_count == 0 or pred_count == 0:
        count_score = 0.0
    else:
        count_score = min(gold_count, pred_count) / max(gold_count, pred_count)

    # Diagnosis match
    gold_diag = _safe_str(getattr(example, "diagnosis", "")).strip().lower()
    pred_diag = _safe_str(getattr(prediction, "diagnosis", "")).strip().lower()
    if gold_diag and pred_diag:
        # Token overlap for partial credit
        gold_tokens = _token_set(gold_diag)
        pred_tokens = _token_set(pred_diag)
        diag_score = len(gold_tokens & pred_tokens) / len(gold_tokens) if gold_tokens else 0.0
    else:
        diag_score = 1.0 if pred_diag else 0.0

    # Malignancy risk match
    gold_risk = _safe_str(getattr(example, "malignancy_risk", "")).strip().lower()
    pred_risk = _safe_str(getattr(prediction, "malignancy_risk", "")).strip().lower()
    risk_score = 1.0 if (gold_risk and gold_risk == pred_risk) else 0.0

    return 0.3 * count_score + 0.4 * diag_score + 0.3 * risk_score
```

Then register it:

```python
"dermatology": dermatology_metric,
```

**`mosaicx/evaluation/optimize.py`** -- add to `_PIPELINE_REGISTRY`:

```python
"dermatology": ("mosaicx.pipelines.dermatology", "DermatologyReportStructurer"),
```

**`mosaicx/cli.py`** -- add `import mosaicx.pipelines.dermatology  # noqa: F401` in the three locations alongside the radiology and pathology imports.

#### Step 4: Create Training Examples

Create `tests/datasets/dermatology_train.jsonl`:

```jsonl
{"document_text": "SKIN BIOPSY REPORT\nSpecimen: Left forearm, punch biopsy\nClinical history: 8mm pigmented lesion, irregular borders\n\nMICROSCOPIC:\nSections show a compound melanocytic nevus with mild architectural disorder. Melanocytes arranged in nests at the dermoepidermal junction. No mitotic figures identified. Margins clear.\n\nDIAGNOSIS: Dysplastic nevus, mild atypia", "lesions": [{"location": "left forearm", "morphology": "compound melanocytic nevus with mild architectural disorder", "size_mm": "8", "margins": "clear"}], "diagnosis": "dysplastic nevus, mild atypia", "malignancy_risk": "low"}
{"document_text": "SKIN BIOPSY REPORT\nSpecimen: Right upper back, excisional biopsy\nClinical history: 12mm erythematous nodule, rapidly growing\n\nMICROSCOPIC:\nSections reveal a well-differentiated squamous cell carcinoma arising in actinic keratosis. Tumor depth 2.1mm. Perineural invasion absent. Deep margin positive.\n\nDIAGNOSIS: Squamous cell carcinoma, well-differentiated", "lesions": [{"location": "right upper back", "morphology": "well-differentiated squamous cell carcinoma arising in actinic keratosis", "size_mm": "12", "margins": "deep margin positive"}], "diagnosis": "squamous cell carcinoma, well-differentiated", "malignancy_risk": "moderate"}
{"document_text": "SKIN BIOPSY REPORT\nSpecimen: Anterior chest, shave biopsy\nClinical history: 5mm dome-shaped papule, skin-colored\n\nMICROSCOPIC:\nSections show an intradermal melanocytic nevus. Melanocytes in well-formed nests within the dermis. No atypia or mitotic activity. Complete excision.\n\nDIAGNOSIS: Intradermal nevus, benign", "lesions": [{"location": "anterior chest", "morphology": "intradermal melanocytic nevus", "size_mm": "5", "margins": "complete excision"}], "diagnosis": "intradermal nevus, benign", "malignancy_risk": "benign"}
```

#### Step 5: Optimize

```bash
mosaicx optimize \
  --pipeline dermatology \
  --trainset tests/datasets/dermatology_train.jsonl \
  --budget light
```

#### Step 6: Test

Run extraction on a single document:

```bash
mosaicx extract --document skin_biopsy.txt --mode dermatology -o result.json
```

Or use the optimized version:

```bash
mosaicx extract \
  --document skin_biopsy.txt \
  --mode dermatology \
  --optimized ~/.mosaicx/optimized/dermatology_optimized.json \
  -o result.json
```

Evaluate against a test set:

```bash
mosaicx eval \
  --pipeline dermatology \
  --testset tests/datasets/dermatology_test.jsonl \
  --optimized ~/.mosaicx/optimized/dermatology_optimized.json
```

---

## Part 2: Python SDK

### Overview

The MOSAICX Python SDK lets you call extraction, de-identification, summarization, and schema generation directly from Python code -- no CLI needed. Every SDK function:

- Accepts plain Python types (strings, dicts, lists, Paths).
- Returns plain Python dicts (not DSPy Predictions or Pydantic models).
- Configures DSPy automatically on first use.
- Supports loading optimized DSPy programs.

For web application integration, the SDK also provides file-based processing (`process_file`, `process_files`) that handles OCR and extraction in a single call, plus `health()` for service monitoring and `list_templates()` for template discovery.

### Installation

```bash
pip install mosaicx
```

For full functionality, ensure you also have DSPy installed:

```bash
pip install mosaicx[all]
```

### Quick Start

```python
from mosaicx.sdk import extract, deidentify, summarize

# Extract structured data from a radiology report
result = extract(
    "CT CHEST WITH CONTRAST\nFindings: 2.3cm RUL nodule...",
    mode="radiology",
)
print(result)
# {"exam_type": "CT Chest", "findings": [...], "impressions": [...], ...}

# Remove PHI from text
clean = deidentify("Patient John Doe, DOB 01/15/1980, SSN 123-45-6789")
print(clean)
# {"redacted_text": "Patient [REDACTED], DOB [REDACTED], SSN [REDACTED]"}

# Summarize multiple reports
summary = summarize([
    "Report 1: CT Chest showing 2.3cm nodule...",
    "Report 2: Follow-up CT showing nodule stable...",
])
print(summary)
# {"narrative": "...", "events": [...]}
```

### API Reference

#### `extract(text, *, template, mode, score, optimized) -> dict`

Extract structured data from document text.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | (required) | The document text to extract from. |
| `template` | `str \| Path \| None` | `None` | Template name (built-in or saved schema), or path to a YAML template file. Resolved via the unified template system. When provided, `mode` is ignored -- the template determines the extraction pipeline. |
| `mode` | `str` | `"auto"` | Extraction mode. `"auto"` lets the LLM infer the schema. `"radiology"` and `"pathology"` run specialized multi-step pipelines. Custom pipelines use their registered name. Ignored when `template` is provided. |
| `score` | `bool` | `False` | If `True`, compute completeness scoring against the template and include it in the output under `"completeness"`. |
| `optimized` | `str \| Path \| None` | `None` | Path to an optimized DSPy program. Only applicable for `mode="auto"` or template-based extraction. |

**Returns:** `dict` -- Structure depends on mode/template. Mode-based extraction includes `_metrics` with timing data. Auto mode may include `inferred_schema`. When `score=True`, includes a `"completeness"` key with coverage metrics.

**Raises:**
- `ValueError` if conflicting parameters are provided, or if the template/mode is unknown.

**Example:**

```python
from mosaicx.sdk import extract

# Auto extraction (LLM infers structure)
result = extract("Patient presents with chest pain and elevated troponin...")
print(result["extracted"])

# Mode-based extraction
result = extract(
    "MRI BRAIN WITHOUT CONTRAST\nFindings: No acute intracranial abnormality.",
    mode="radiology",
)
print(result["exam_type"])    # "MRI Brain"
print(result["findings"])     # [...]

# Template-based extraction
result = extract(
    "Lab results: WBC 12.3, Hgb 10.1, Plt 245",
    template="lab_results",
)
print(result["extracted"])

# Template-based extraction with completeness scoring
result = extract(
    "CT CHEST WITH CONTRAST\nFindings: 2.3cm RUL nodule...",
    template="chest_ct",
    score=True,
)
print(result["extracted"])
print(result["completeness"])  # {"overall": 0.85, "missing_required": [...], ...}

# With an optimized program
result = extract(
    "CT ABDOMEN...",
    mode="auto",
    optimized="~/.mosaicx/optimized/extract_optimized.json",
)
```

---

#### `report(text, *, template, schema_name, describe, mode) -> dict` -- DEPRECATED

> **Deprecated.** Use `extract(score=True)` instead. This function still works but emits a `DeprecationWarning`.

Extract structured data and score completeness against a template. Internally delegates to `extract()` with `score=True`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | (required) | Document text to extract from. |
| `template` | `str \| Path \| None` | `None` | Built-in RDES template name (e.g., `"chest_ct"`) or path to a YAML template file. |
| `schema_name` | `str \| None` | `None` | *Deprecated* -- use `template` instead. |
| `describe` | `str \| None` | `None` | No longer supported. Use `mosaicx template create --describe` to create a template first, then pass the template name. |
| `mode` | `str \| None` | `None` | Explicit pipeline mode override. If `None`, auto-detected from the template. |

**Returns:** `dict` with keys `"extracted"`, `"completeness"`, `"template_name"`, `"mode_used"`, `"metrics"`.

**Migration example:**

```python
# Before (deprecated)
from mosaicx.sdk import report
result = report("CT Chest report text...", template="chest_ct")

# After (preferred)
from mosaicx.sdk import extract
result = extract("CT Chest report text...", template="chest_ct", score=True)
```

---

#### `deidentify(text, *, mode) -> dict`

Remove Protected Health Information (PHI) from text.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | (required) | Text containing PHI. |
| `mode` | `str` | `"remove"` | De-identification strategy: `"remove"` replaces PHI with `[REDACTED]`. `"pseudonymize"` replaces with realistic fake values. `"dateshift"` shifts dates by a consistent random offset. `"regex"` uses regex-only scrubbing (no LLM needed). |

**Returns:** `dict` with key `"redacted_text"` (str).

**Raises:** `ValueError` if `mode` is not one of the supported values.

**Example:**

```python
from mosaicx.sdk import deidentify

# Default: replace PHI with [REDACTED]
result = deidentify("Patient John Doe, SSN 123-45-6789")
print(result["redacted_text"])
# "Patient [REDACTED], SSN [REDACTED]"

# Pseudonymize: replace with fake values
result = deidentify(
    "Patient Jane Smith, DOB 03/22/1975",
    mode="pseudonymize",
)
print(result["redacted_text"])
# "Patient Maria Garcia, DOB 07/14/1982"

# Regex-only: no LLM call, fastest option
result = deidentify("SSN: 123-45-6789, MRN: 12345678", mode="regex")
print(result["redacted_text"])
```

---

#### `summarize(reports, *, patient_id, optimized) -> dict`

Summarize multiple clinical reports into a patient timeline.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reports` | `list[str]` | (required) | List of report texts. |
| `patient_id` | `str` | `"unknown"` | Patient identifier for the summary. |
| `optimized` | `str \| Path \| None` | `None` | Path to an optimized DSPy program. |

**Returns:** `dict` with keys `"narrative"` (str) and `"events"` (list of event dicts).

**Raises:** `ValueError` if `reports` is empty.

**Example:**

```python
from mosaicx.sdk import summarize

reports = [
    "2024-01-15: CT Chest showing 2.3cm RUL nodule. Recommend follow-up.",
    "2024-04-20: Follow-up CT: RUL nodule stable at 2.3cm. Continue surveillance.",
    "2024-10-10: PET/CT: RUL nodule with low-grade uptake. SUVmax 2.1.",
]

result = summarize(reports, patient_id="PAT-001")
print(result["narrative"])
# "Patient PAT-001 was found to have a 2.3cm right upper lobe nodule..."
print(result["events"])
# [{"date": "2024-01-15", "event": "Initial CT...", ...}, ...]
```

---

#### `generate_schema(description, *, name, example_text, save) -> dict`

Generate a Pydantic schema from a plain-English description. For most use cases, the CLI command `mosaicx template create` is the preferred way to create templates, since it integrates with the unified template system. This SDK function remains available for programmatic schema generation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `description` | `str` | (required) | Natural language description of desired fields. |
| `name` | `str \| None` | `None` | Optional schema name. If omitted the LLM will choose one. |
| `example_text` | `str \| None` | `None` | Optional example document text to guide schema generation. |
| `save` | `bool` | `False` | If `True`, persist the schema to `~/.mosaicx/schemas/`. |

**Returns:** `dict` with keys `"name"` (str), `"fields"` (list of field dicts), `"json_schema"` (dict). If `save=True`, also includes `"saved_to"` (str).

**Example:**

```python
from mosaicx.sdk import generate_schema

schema = generate_schema(
    "Echocardiogram report with LVEF, chamber dimensions, valve grades",
    name="EchoReport",
    save=True,
)
print(schema["name"])        # "EchoReport"
print(schema["fields"])      # [{"name": "lvef", "type": "str", ...}, ...]
print(schema["saved_to"])    # "~/.mosaicx/schemas/EchoReport.json"
```

---

#### `list_schemas() -> list[str]`

List names of all saved schemas.

**Returns:** `list[str]` -- Schema names, alphabetically sorted. Empty list if no schemas exist.

**Example:**

```python
from mosaicx.sdk import list_schemas

schemas = list_schemas()
print(schemas)
# ["EchoReport", "LabResults", "OperativeNote"]
```

---

#### `list_modes() -> list[dict]`

List available extraction modes with descriptions.

**Returns:** `list[dict]` -- Each dict has keys `"name"` and `"description"`.

**Example:**

```python
from mosaicx.sdk import list_modes

modes = list_modes()
for m in modes:
    print(f"{m['name']}: {m['description']}")
# radiology: 5-step radiology report structurer (findings, measurements, scoring)
# pathology: ...
```

---

#### `evaluate(pipeline, testset_path, *, optimized) -> dict`

Evaluate a pipeline against a labeled test set.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | `str` | (required) | Pipeline name (e.g., `"radiology"`, `"pathology"`, `"extract"`). |
| `testset_path` | `str \| Path` | (required) | Path to a `.jsonl` file with labeled examples. |
| `optimized` | `str \| Path \| None` | `None` | Path to an optimized DSPy program. If `None`, evaluates the baseline. |

**Returns:** `dict` with keys `"mean"`, `"median"`, `"std"` (or `None` if fewer than 2 examples), `"min"`, `"max"`, `"count"`, `"scores"` (list of floats).

**Raises:**
- `ValueError` if `pipeline` is not recognized.
- `FileNotFoundError` if `testset_path` does not exist.

**Example:**

```python
from mosaicx.sdk import evaluate

# Evaluate baseline
results = evaluate("radiology", "tests/datasets/radiology_test.jsonl")
print(f"Mean score: {results['mean']:.3f}")
print(f"Scores: {results['scores']}")

# Evaluate optimized
results = evaluate(
    "radiology",
    "tests/datasets/radiology_test.jsonl",
    optimized="~/.mosaicx/optimized/radiology_optimized.json",
)
print(f"Optimized mean: {results['mean']:.3f}")
```

---

#### `batch_extract(texts, *, mode, schema_name) -> list[dict]`

Extract structured data from multiple documents. A convenience wrapper that calls `extract()` for each text.

> **Note:** The `schema_name` parameter on `batch_extract()` still uses the legacy name. For template-based batch extraction, call `extract()` with the `template` parameter in a loop instead.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | `list[str]` | (required) | List of document texts. |
| `mode` | `str` | `"auto"` | Extraction mode (same as `extract()`). |
| `schema_name` | `str \| None` | `None` | Name of a saved schema. For template-based extraction, use `extract()` with `template` directly. |

**Returns:** `list[dict]` -- One result dict per input text. Failed extractions produce a dict with an `"error"` key.

**Example:**

```python
from mosaicx.sdk import batch_extract

texts = [
    "CT Chest: Normal lung fields...",
    "MRI Brain: No acute findings...",
    "X-ray Knee: Mild degenerative changes...",
]

results = batch_extract(texts, mode="radiology")
for i, result in enumerate(results):
    if "error" in result:
        print(f"Document {i} failed: {result['error']}")
    else:
        print(f"Document {i}: {result['exam_type']}")
```

---

#### `health() -> dict`

Check MOSAICX configuration status and available capabilities. Does not make any LLM calls -- useful for service health endpoints.

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `version` | `str` | Installed MOSAICX version. |
| `configured` | `bool` | Whether an API key is set. |
| `lm_model` | `str` | Configured language model identifier. |
| `api_base` | `str` | LLM API base URL. |
| `available_modes` | `list[str]` | Registered extraction modes (e.g., `["radiology", "pathology"]`). |
| `available_templates` | `list[str]` | Available template names (built-in + user-created). |
| `ocr_engine` | `str` | Configured OCR engine (`"both"`, `"surya"`, or `"chandra"`). |

**Example:**

```python
from mosaicx.sdk import health

status = health()
print(f"MOSAICX v{status['version']}")
print(f"LLM: {status['lm_model']} at {status['api_base']}")
print(f"Modes: {status['available_modes']}")
print(f"Templates: {status['available_templates']}")
```

---

#### `list_templates() -> list[dict]`

List all available extraction templates -- both built-in templates that ship with MOSAICX and user-created templates from `~/.mosaicx/templates/`.

**Returns:** `list[dict]` -- Each dict has keys:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Template name (use with `extract(template=name)`). |
| `description` | `str` | Human-readable description. |
| `mode` | `str \| None` | Associated pipeline mode (e.g., `"radiology"`), or `None`. |
| `source` | `str` | `"built-in"` or `"user"`. |

**Example:**

```python
from mosaicx.sdk import list_templates

for tpl in list_templates():
    print(f"{tpl['name']:20s} [{tpl['source']}] {tpl['description']}")
```

---

#### `process_file(file, *, filename, template, mode, score, ocr_engine, force_ocr) -> dict`

Load a document and extract structured data in one call. Handles OCR for PDFs and images, then runs the extraction pipeline. Accepts a file path or raw bytes (e.g., from a web file upload).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | `Path \| bytes` | (required) | Path to a document file, or raw file bytes. |
| `filename` | `str \| None` | `None` | Original filename. Required when `file` is bytes (for format detection). |
| `template` | `str \| None` | `None` | Template name or YAML file path. |
| `mode` | `str` | `"auto"` | Extraction mode (`"auto"`, `"radiology"`, `"pathology"`). |
| `score` | `bool` | `False` | Include completeness scoring. |
| `ocr_engine` | `str \| None` | `None` | Override configured OCR engine. |
| `force_ocr` | `bool` | `False` | Force OCR even on PDFs with native text. |

**Returns:** `dict` -- Extraction result (same as `extract()`), plus a `"_document"` key with loading metadata:

```python
{
    "extracted": {...},          # structured data
    "_document": {
        "format": "pdf",
        "page_count": 3,
        "ocr_engine_used": "surya",
        "quality_warning": False,
    }
}
```

**Raises:**
- `ValueError` if `file` is bytes and `filename` is not provided.
- `FileNotFoundError` if `file` is a path that does not exist.

**Example -- file path:**

```python
from pathlib import Path
from mosaicx.sdk import process_file

result = process_file(
    Path("scan.pdf"),
    template="chest_ct",
    score=True,
)
print(result["extracted"])
print(f"Pages: {result['_document']['page_count']}")
```

**Example -- bytes from a web upload:**

```python
from mosaicx.sdk import process_file

# In a web handler (FastAPI, Flask, Django, etc.)
content = uploaded_file.read()
result = process_file(
    content,
    filename=uploaded_file.filename,  # needed for format detection
    template="chest_ct",
)
```

---

#### `process_files(files, *, template, mode, score, workers, on_progress) -> dict`

Process multiple documents with parallel extraction. Accepts a directory path (discovers all supported files) or an explicit list of file paths. Documents are loaded sequentially (OCR is not thread-safe), but extraction runs in parallel.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `files` | `list[Path] \| Path` | (required) | Directory path or list of file paths. |
| `template` | `str \| None` | `None` | Template name for targeted extraction. |
| `mode` | `str` | `"auto"` | Extraction mode. |
| `score` | `bool` | `False` | Include completeness scoring. |
| `workers` | `int` | `4` | Number of parallel extraction workers (max 32). |
| `on_progress` | `Callable \| None` | `None` | Callback `(filename, success, result_or_none)` after each document. |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `total` | `int` | Total documents discovered. |
| `succeeded` | `int` | Successfully extracted. |
| `failed` | `int` | Failed (load error, extraction error, or empty text). |
| `results` | `list[dict]` | Successful extraction results (each includes a `"file"` key). |
| `errors` | `list[dict]` | Failed documents (each has `"file"` and `"error"` keys). |

**Example -- directory:**

```python
from pathlib import Path
from mosaicx.sdk import process_files

results = process_files(
    Path("reports/"),
    template="chest_ct",
    workers=8,
)
print(f"{results['succeeded']}/{results['total']} succeeded")
for r in results["results"]:
    print(f"  {r['file']}: {r.get('exam_type', 'auto')}")
for e in results["errors"]:
    print(f"  FAILED {e['file']}: {e['error']}")
```

**Example -- with progress callback (for web apps):**

```python
from mosaicx.sdk import process_files

def on_progress(filename, success, result):
    status = "done" if success else "FAILED"
    print(f"[{status}] {filename}")
    # In a web app, send this to a WebSocket or SSE stream

results = process_files(
    file_list,
    template="chest_ct",
    on_progress=on_progress,
)
```

### Configuration

The SDK uses the same configuration system as the CLI. Configuration is resolved in this order (highest priority first):

1. Environment variables with the `MOSAICX_` prefix
2. Values in a `.env` file in the current directory
3. Built-in defaults

Set environment variables before calling SDK functions:

```bash
export MOSAICX_LM="openai/gpt-oss:120b"
export MOSAICX_API_BASE="http://localhost:11434/v1"
export MOSAICX_API_KEY="your-api-key"
```

Or use a `.env` file in your project root:

```ini
MOSAICX_LM=openai/gpt-oss:120b
MOSAICX_API_BASE=http://localhost:11434/v1
MOSAICX_API_KEY=your-api-key
MOSAICX_OCR_ENGINE=both
```

Or set them in Python before your first SDK call:

```python
import os
os.environ["MOSAICX_LM"] = "openai/gpt-oss:120b"
os.environ["MOSAICX_API_BASE"] = "http://localhost:11434/v1"
os.environ["MOSAICX_API_KEY"] = "your-api-key"

from mosaicx.sdk import extract
result = extract("Patient presents with...")
```

DSPy is configured automatically on the first SDK call that needs it. The `deidentify()` function with `mode="regex"` does not require DSPy or an API key.

### Using Optimized Programs

Every extraction and summarization function accepts an `optimized` parameter pointing to a saved DSPy program. Optimized programs contain tuned prompts and few-shot demonstrations that improve accuracy.

```python
from mosaicx.sdk import extract, summarize

# Use baseline (no optimization)
result = extract("CT Chest...", mode="auto")

# Use optimized program
result = extract(
    "CT Chest...",
    mode="auto",
    optimized="~/.mosaicx/optimized/extract_optimized.json",
)

# Summarize with optimized program
summary = summarize(
    ["Report 1...", "Report 2..."],
    optimized="~/.mosaicx/optimized/summarize_optimized.json",
)
```

To create an optimized program, use the CLI:

```bash
mosaicx optimize --pipeline extract --trainset data/train.jsonl --budget medium
```

Or use the SDK's `evaluate()` to measure improvement:

```python
from mosaicx.sdk import evaluate

baseline = evaluate("extract", "data/test.jsonl")
optimized = evaluate(
    "extract",
    "data/test.jsonl",
    optimized="~/.mosaicx/optimized/extract_optimized.json",
)

print(f"Baseline: {baseline['mean']:.3f}")
print(f"Optimized: {optimized['mean']:.3f}")
```

### Error Handling

SDK functions raise standard Python exceptions. Here are the most common ones and how to handle them.

#### RuntimeError: No API Key

```python
from mosaicx.sdk import extract

try:
    result = extract("some text")
except RuntimeError as e:
    if "No API key" in str(e):
        print("Set MOSAICX_API_KEY before calling SDK functions")
    elif "DSPy is required" in str(e):
        print("Install DSPy: pip install dspy")
```

#### ValueError: Unknown Mode or Pipeline

```python
from mosaicx.sdk import extract, evaluate

try:
    result = extract("some text", mode="nonexistent")
except ValueError as e:
    print(f"Invalid mode: {e}")

try:
    result = evaluate("nonexistent", "data/test.jsonl")
except ValueError as e:
    print(f"Invalid pipeline: {e}")
```

#### FileNotFoundError: Missing Template or Dataset

```python
from mosaicx.sdk import extract, evaluate

try:
    result = extract("some text", template="nonexistent_template")
except FileNotFoundError as e:
    print(f"Template not found: {e}")

try:
    result = evaluate("radiology", "nonexistent_file.jsonl")
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")
```

#### Batch Processing with Error Recovery

The `batch_extract()` function catches exceptions per-document and returns `{"error": "..."}` instead of failing the entire batch:

```python
from mosaicx.sdk import batch_extract

results = batch_extract(["valid text", "", "another valid text"])
for i, result in enumerate(results):
    if "error" in result:
        print(f"Document {i} failed: {result['error']}")
    else:
        print(f"Document {i} extracted successfully")
```

### Integration Examples

#### Batch Processing with pandas

```python
import pandas as pd
from mosaicx.sdk import extract

# Load documents from a CSV
df = pd.read_csv("reports.csv")

# Extract from each row
results = []
for _, row in df.iterrows():
    try:
        result = extract(row["report_text"], mode="radiology")
        result["source_id"] = row["report_id"]
        results.append(result)
    except Exception as e:
        results.append({"source_id": row["report_id"], "error": str(e)})

# Flatten nested results into a DataFrame
results_df = pd.json_normalize(results, sep="_")
results_df.to_csv("extracted_results.csv", index=False)
results_df.to_parquet("extracted_results.parquet", index=False)
```

#### Using in a FastAPI Endpoint

```python
import os
os.environ["MOSAICX_API_KEY"] = "your-api-key"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mosaicx.sdk import extract, deidentify, list_modes

app = FastAPI(title="MOSAICX API")


class ExtractRequest(BaseModel):
    text: str
    template: str | None = None
    mode: str = "auto"
    score: bool = False


class DeidentifyRequest(BaseModel):
    text: str
    mode: str = "remove"


@app.get("/modes")
def get_modes():
    return list_modes()


@app.post("/extract")
def run_extract(req: ExtractRequest):
    try:
        return extract(
            req.text,
            template=req.template,
            mode=req.mode,
            score=req.score,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deidentify")
def run_deidentify(req: DeidentifyRequest):
    try:
        return deidentify(req.text, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

Run with:

```bash
uvicorn myapp:app --reload
```

#### Jupyter Notebook Usage

```python
# Cell 1: Configure
import os
os.environ["MOSAICX_LM"] = "openai/gpt-oss:120b"
os.environ["MOSAICX_API_BASE"] = "http://localhost:11434/v1"
os.environ["MOSAICX_API_KEY"] = "your-api-key"
```

```python
# Cell 2: Extract from a report
from mosaicx.sdk import extract

report = """
CT CHEST WITH CONTRAST
Clinical indication: Cough, weight loss

FINDINGS:
Right upper lobe: 2.3 cm spiculated nodule (series 4, image 67).
Left lung: Clear.
Mediastinum: No lymphadenopathy.
Pleura: No effusion.

IMPRESSION:
1. 2.3 cm spiculated RUL nodule, suspicious for malignancy.
   Recommend PET/CT and tissue sampling.
"""

result = extract(report, mode="radiology")
result
```

```python
# Cell 3: Explore the results
import json
print(json.dumps(result, indent=2, default=str))

# Access specific fields
print(f"Exam type: {result.get('exam_type')}")
print(f"Number of findings: {len(result.get('findings', []))}")
```

```python
# Cell 4: Process multiple reports
from mosaicx.sdk import batch_extract
import pandas as pd

reports = [
    "CT Head: No acute intracranial abnormality.",
    "MRI Spine: L4-L5 disc herniation with moderate canal stenosis.",
    "X-ray Chest: Bilateral lower lobe opacities, concerning for pneumonia.",
]

results = batch_extract(reports, mode="radiology")
df = pd.json_normalize(results, sep="_")
df
```

```python
# Cell 5: De-identify before sharing
from mosaicx.sdk import deidentify

sensitive = "Patient John Smith, MRN 12345, DOB 01/15/1960"
clean = deidentify(sensitive)
print(clean["redacted_text"])
```

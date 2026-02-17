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

For the complete SDK reference -- function signatures, parameter tables, input/output matrix, and usage examples -- see the dedicated [SDK Reference](sdk-reference.md).

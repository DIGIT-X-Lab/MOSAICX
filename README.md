# MOSAICX

**Medical cOmputational Suite for Advanced Intelligent eXtraction**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

MOSAICX is an intelligent radiology report extraction tool that uses local Large Language Models (LLMs) to extract structured data from medical reports. It supports both PDF and text inputs, provides configurable output formats, and offers both programmatic and command-line interfaces.

## Features

🔬 **Intelligent Extraction**: Uses local LLMs (Ollama) for context-aware data extraction  
📄 **Advanced Document Processing**: Powered by Docling for superior PDF and document parsing  
⚙️ **Configurable Schemas**: Define custom extraction schemas with interactive brainstorming  
📊 **Flexible Outputs**: Export to JSON, CSV, or custom formats  
🔄 **Multi-Report Analysis**: Process multiple reports for patient history synthesis  
🖥️ **Dual Interface**: Use as Python library or CLI tool  
🏠 **Local Processing**: All processing happens locally using Ollama - no cloud dependencies  
⚡ **Fast Development**: Built with uv for lightning-fast dependency management  

## Quick Start

### Installation

```bash
pip install mosaicx
```

**For Development (with uv - recommended):**

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/LalithShiyam/MOSAICX.git
cd MOSAICX
uv sync --dev
uv run pre-commit install
```

### Basic Usage

#### Command Line Interface

```bash
# Extract from a single PDF report  
uv run mosaicx extract report.pdf --config extraction_config.yaml --output results.json

# Interactive schema building
uv run mosaicx brainstorm --report sample_report.pdf --schema-output custom_schema.yaml

# Batch processing multiple reports
uv run mosaicx extract-batch reports/ --config config.yaml --output-dir results/
```

#### Python Library

```python
from mosaicx import ReportExtractor, ExtractionConfig

# Initialize extractor
extractor = ReportExtractor()

# Extract from PDF
config = ExtractionConfig.from_file('config.yaml')
results = extractor.extract_from_pdf('report.pdf', config)

# Extract from text
text_content = "Patient shows signs of pneumonia..."
results = extractor.extract_from_text(text_content, config)

# Multi-report analysis
patient_reports = ['report1.pdf', 'report2.pdf', 'report3.pdf']
timeline = extractor.analyze_patient_history(patient_reports, config)
```

## Configuration

Create a YAML configuration file to define extraction schemas:

```yaml
schema:
  findings:
    - field: "primary_diagnosis"
      type: "string"
      description: "Main diagnosis from the report"
    - field: "severity"
      type: "enum"
      options: ["mild", "moderate", "severe"]
    - field: "follow_up_required"
      type: "boolean"

output:
  format: "json"
  include_confidence: true
  include_source_text: true

llm:
  model: "llama2"
  temperature: 0.1
  max_tokens: 1000
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Examples](examples/)

## Development

MOSAICX is developed by the DIGITX Lab at the Department of Radiology, LMU Munich University Hospital.

### Requirements

- Python 3.11+
- Ollama installed locally
- Local LLM model (e.g., Llama2, CodeLlama)

### Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Authors

**Lalith Kumar Shiyam Sundar, PhD**  
DIGITX Lab, Department of Radiology  
LMU Munich University Hospital  
📧 lalith.shiyam@med.uni-muenchen.de

## Citation

If you use MOSAICX in your research, please cite:

```bibtex
@software{mosaicx2024,
  title={MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction},
  author={Sundar, Lalith Kumar Shiyam},
  year={2024},
  institution={DIGITX Lab, Department of Radiology, LMU Munich University Hospital},
  url={https://github.com/LalithShiyam/MOSAICX}
}
```

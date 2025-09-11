"""Contract tests for the MOSAICX library API.

These tests define the expected behavior of the Python library interface
before implementation. They serve as the contract that the implementation
must fulfill.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from mosaicx import ReportExtractor, ExtractionConfig, AnalysisConfig
from mosaicx.core.models import ExtractionResult, AnalysisResult
from mosaicx.core.exceptions import (
    ExtractionError,
    ConfigurationError,
    ModelError,
    ValidationError as MosaicxValidationError
)


class TestReportExtractorAPI:
    """Contract tests for the ReportExtractor class."""

    @pytest.fixture
    def sample_text_content(self) -> str:
        """Sample medical report text."""
        return """
RADIOLOGY REPORT

Patient: Sarah Johnson
DOB: 1975-08-12
MRN: 987654321
Date: 2024-01-25
Study: CT Chest with Contrast

CLINICAL HISTORY:
52-year-old female with persistent cough and weight loss.

TECHNIQUE:
Axial CT images of the chest obtained with IV contrast.

FINDINGS:
1. Large mass in the left upper lobe measuring 4.2 x 3.8 cm
2. Multiple enlarged mediastinal lymph nodes
3. Small left pleural effusion
4. No evidence of osseous metastases

IMPRESSION:
1. Left upper lobe mass with mediastinal lymphadenopathy
2. Highly suspicious for primary lung malignancy
3. Small pleural effusion
4. Recommend tissue sampling and staging workup

Dr. Michael Chen, MD
Signed: 2024-01-25 16:45:00
"""

    @pytest.fixture
    def sample_pdf_path(self, tmp_path: Path, sample_text_content: str) -> Path:
        """Create a sample PDF file for testing."""
        pdf_path = tmp_path / "sample_report.pdf"
        # Create a minimal PDF structure with text content
        pdf_content = f"""%PDF-1.4
1 0 obj
<</Type/Catalog/Pages 2 0 R>>
endobj
2 0 obj  
<</Type/Pages/Kids[3 0 R]/Count 1>>
endobj
3 0 obj
<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>
endobj
4 0 obj
<</Length {len(sample_text_content)}>>
stream
{sample_text_content}
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000015 00000 n 
0000000060 00000 n 
0000000120 00000 n 
0000000203 00000 n 
trailer
<</Size 5/Root 1 0 R>>
startxref
{300 + len(sample_text_content)}
%%EOF"""
        pdf_path.write_text(pdf_content)
        return pdf_path

    @pytest.fixture
    def sample_extraction_config(self, tmp_path: Path) -> Path:
        """Create a sample extraction configuration."""
        config_path = tmp_path / "extraction_config.yaml"
        config_content = """
schema:
  findings:
    - field: "patient_name"
      type: "string"
      description: "Patient's full name"
    - field: "mrn"
      type: "string"
      description: "Medical record number"
    - field: "study_date"
      type: "date"
      description: "Date of the study"
    - field: "study_type"
      type: "string"
      description: "Type of imaging study"
    - field: "primary_findings"
      type: "array"
      description: "List of key findings"
    - field: "impression"
      type: "string"
      description: "Radiologist's impression"
    - field: "recommendations"
      type: "string"
      description: "Follow-up recommendations"

validation:
  required_fields:
    - "patient_name"
    - "study_date"
    - "impression"
  
output:
  format: "json"
  include_confidence: true
  include_source_text: true
  confidence_threshold: 0.7

llm:
  model: "llama3"
  temperature: 0.1
  max_tokens: 1500
"""
        config_path.write_text(config_content)
        return config_path

    def test_report_extractor_initialization(self) -> None:
        """Test ReportExtractor can be initialized with default settings."""
        extractor = ReportExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'extract_from_text')
        assert hasattr(extractor, 'extract_from_pdf')
        assert hasattr(extractor, 'extract_from_file')

    def test_report_extractor_custom_initialization(self) -> None:
        """Test ReportExtractor initialization with custom parameters."""
        extractor = ReportExtractor(
            model_name="llama3",
            temperature=0.2,
            max_tokens=2000,
            ollama_url="http://localhost:11434"
        )
        assert extractor is not None

    def test_extraction_config_from_dict(self) -> None:
        """Test creating ExtractionConfig from dictionary."""
        config_dict = {
            "schema": {
                "findings": [
                    {
                        "field": "patient_name",
                        "type": "string",
                        "description": "Patient's name"
                    }
                ]
            },
            "output": {"format": "json"},
            "llm": {"model": "llama2"}
        }
        
        config = ExtractionConfig.from_dict(config_dict)
        assert config is not None
        assert config.schema is not None

    def test_extraction_config_from_file(self, sample_extraction_config: Path) -> None:
        """Test creating ExtractionConfig from YAML file."""
        config = ExtractionConfig.from_file(sample_extraction_config)
        assert config is not None
        assert config.schema is not None
        assert len(config.schema.findings) > 0

    def test_extraction_config_validation(self) -> None:
        """Test ExtractionConfig validation for invalid configuration."""
        with pytest.raises((ValidationError, MosaicxValidationError)):
            # Missing required schema
            ExtractionConfig.from_dict({
                "output": {"format": "json"}
            })

    def test_extract_from_text_success(
        self, 
        sample_text_content: str,
        sample_extraction_config: Path
    ) -> None:
        """Test successful text extraction."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        result = extractor.extract_from_text(sample_text_content, config)
        
        assert isinstance(result, ExtractionResult)
        assert result.extracted_data is not None
        assert result.metadata is not None
        assert result.confidence_scores is not None
        assert "patient_name" in result.extracted_data
        assert "study_date" in result.extracted_data

    def test_extract_from_pdf_success(
        self,
        sample_pdf_path: Path,
        sample_extraction_config: Path
    ) -> None:
        """Test successful PDF extraction."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        result = extractor.extract_from_pdf(sample_pdf_path, config)
        
        assert isinstance(result, ExtractionResult)
        assert result.extracted_data is not None
        assert result.source_file == str(sample_pdf_path)

    def test_extract_from_file_auto_detection(
        self,
        sample_pdf_path: Path,
        sample_extraction_config: Path
    ) -> None:
        """Test automatic file type detection."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        result = extractor.extract_from_file(sample_pdf_path, config)
        
        assert isinstance(result, ExtractionResult)
        assert result.extracted_data is not None

    def test_extract_missing_file_error(self, sample_extraction_config: Path) -> None:
        """Test error handling for missing files."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        with pytest.raises(ExtractionError):
            extractor.extract_from_file(Path("nonexistent_file.pdf"), config)

    def test_extract_invalid_config_error(self, sample_text_content: str) -> None:
        """Test error handling for invalid configuration."""
        extractor = ReportExtractor()
        
        with pytest.raises((ConfigurationError, ValidationError)):
            # Invalid config should raise error
            invalid_config = ExtractionConfig.from_dict({})
            extractor.extract_from_text(sample_text_content, invalid_config)

    def test_extraction_result_properties(
        self,
        sample_text_content: str,
        sample_extraction_config: Path
    ) -> None:
        """Test ExtractionResult properties and methods."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        result = extractor.extract_from_text(sample_text_content, config)
        
        # Test properties
        assert hasattr(result, 'extracted_data')
        assert hasattr(result, 'confidence_scores')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'model_info')
        
        # Test methods
        assert hasattr(result, 'to_json')
        assert hasattr(result, 'to_dict')
        assert hasattr(result, 'get_high_confidence_fields')
        
        # Test serialization
        json_output = result.to_json()
        assert isinstance(json_output, str)
        json.loads(json_output)  # Should be valid JSON
        
        dict_output = result.to_dict()
        assert isinstance(dict_output, dict)

    def test_confidence_filtering(
        self,
        sample_text_content: str,
        sample_extraction_config: Path
    ) -> None:
        """Test confidence-based result filtering."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        result = extractor.extract_from_text(sample_text_content, config)
        
        high_confidence_fields = result.get_high_confidence_fields(threshold=0.8)
        assert isinstance(high_confidence_fields, dict)
        
        # All returned fields should meet confidence threshold
        for field_name in high_confidence_fields:
            if field_name in result.confidence_scores:
                assert result.confidence_scores[field_name] >= 0.8

    def test_batch_processing(
        self,
        tmp_path: Path,
        sample_extraction_config: Path
    ) -> None:
        """Test batch processing of multiple files."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        # Create multiple test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"report_{i}.txt"
            file_path.write_text(f"Patient: Test Patient {i}\nFindings: Normal")
            files.append(file_path)
        
        results = extractor.extract_batch(files, config)
        
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ExtractionResult)

    def test_async_extraction(
        self,
        sample_text_content: str,
        sample_extraction_config: Path
    ) -> None:
        """Test asynchronous extraction capability."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        # Test async method exists and returns awaitable
        async_result = extractor.extract_from_text_async(sample_text_content, config)
        assert hasattr(async_result, '__await__')

    def test_model_configuration_override(
        self,
        sample_text_content: str,
        sample_extraction_config: Path
    ) -> None:
        """Test overriding model configuration at runtime."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        result = extractor.extract_from_text(
            sample_text_content, 
            config,
            model_override="llama3",
            temperature_override=0.5
        )
        
        assert isinstance(result, ExtractionResult)
        assert result.model_info["model"] == "llama3"
        assert result.model_info["temperature"] == 0.5

    def test_custom_schema_validation(
        self,
        sample_text_content: str
    ) -> None:
        """Test custom schema validation and field types."""
        custom_config_dict = {
            "schema": {
                "findings": [
                    {
                        "field": "patient_name",
                        "type": "string",
                        "required": True,
                        "validation": {
                            "min_length": 2,
                            "pattern": r"^[A-Za-z\s]+$"
                        }
                    },
                    {
                        "field": "age",
                        "type": "integer",
                        "validation": {
                            "min_value": 0,
                            "max_value": 150
                        }
                    },
                    {
                        "field": "study_date",
                        "type": "date",
                        "required": True
                    }
                ]
            },
            "output": {"format": "json"},
            "llm": {"model": "llama2"}
        }
        
        extractor = ReportExtractor()
        config = ExtractionConfig.from_dict(custom_config_dict)
        
        result = extractor.extract_from_text(sample_text_content, config)
        assert isinstance(result, ExtractionResult)

    def test_progress_callback(
        self,
        tmp_path: Path,
        sample_extraction_config: Path
    ) -> None:
        """Test progress callback functionality for batch processing."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        progress_calls = []
        def progress_callback(current: int, total: int, current_file: str) -> None:
            progress_calls.append((current, total, current_file))
        
        # Create test files
        files = []
        for i in range(3):
            file_path = tmp_path / f"report_{i}.txt"
            file_path.write_text(f"Patient: Test Patient {i}")
            files.append(file_path)
        
        results = extractor.extract_batch(
            files, 
            config, 
            progress_callback=progress_callback
        )
        
        assert len(results) == 3
        assert len(progress_calls) > 0

    def test_error_handling_modes(
        self,
        tmp_path: Path,
        sample_extraction_config: Path
    ) -> None:
        """Test different error handling modes during batch processing."""
        extractor = ReportExtractor()
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        # Create mix of valid and invalid files
        files = []
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("Patient: Valid Patient")
        files.append(valid_file)
        
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("")  # Empty file
        files.append(invalid_file)
        
        # Test with skip_errors=True
        results = extractor.extract_batch(
            files, 
            config, 
            skip_errors=True
        )
        
        assert len(results) >= 1  # Should have at least one successful result
        
        # Test with skip_errors=False should raise exception
        with pytest.raises(ExtractionError):
            extractor.extract_batch(
                files, 
                config, 
                skip_errors=False
            )

    def test_caching_functionality(
        self,
        sample_text_content: str,
        sample_extraction_config: Path
    ) -> None:
        """Test result caching functionality."""
        extractor = ReportExtractor(enable_cache=True)
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        # First extraction
        result1 = extractor.extract_from_text(sample_text_content, config)
        
        # Second extraction of same content should use cache
        result2 = extractor.extract_from_text(sample_text_content, config)
        
        assert result1.extracted_data == result2.extracted_data
        # Second call should be much faster (cached)
        assert result2.metadata.get("cached", False) == True

    def test_memory_management(
        self,
        tmp_path: Path,
        sample_extraction_config: Path
    ) -> None:
        """Test memory management during large batch processing."""
        extractor = ReportExtractor(max_memory_mb=100)
        config = ExtractionConfig.from_file(sample_extraction_config)
        
        # Create many files to test memory limits
        files = []
        for i in range(10):
            file_path = tmp_path / f"report_{i}.txt"
            # Create larger content to test memory management
            content = f"Patient: Test Patient {i}\n" + "Test content " * 1000
            file_path.write_text(content)
            files.append(file_path)
        
        # Should complete without memory errors
        results = extractor.extract_batch(files, config)
        assert len(results) == 10

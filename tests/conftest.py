"""
Test Configuration and Fixtures

================================================================================
MOSAICX Test Suite Configuration
================================================================================

This module provides common test fixtures, utilities, and configuration
for the MOSAICX test suite. It includes pytest fixtures for temporary
files, mock data, and test environment setup.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

# Sample test data
SAMPLE_PDF_CONTENT = "This is sample PDF content for testing extraction."

SAMPLE_PATIENT_DATA = {
    "patient_id": "P123456",
    "name": "John Doe",
    "age": 45,
    "gender": "Male",
    "diagnosis": "Hypertension"
}

SAMPLE_SCHEMA_DESCRIPTION = "Patient demographics with name, age, gender, and diagnosis"

SAMPLE_PYDANTIC_CODE = '''
from pydantic import BaseModel, Field
from typing import Optional

class PatientRecord(BaseModel):
    """Patient demographic and clinical information."""
    
    patient_id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., description="Patient full name")
    age: int = Field(..., ge=0, le=150, description="Patient age in years")
    gender: str = Field(..., description="Patient gender")
    diagnosis: Optional[str] = Field(None, description="Primary diagnosis")
'''

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_pdf_file(temp_dir):
    """Create a temporary PDF file for testing."""
    pdf_file = temp_dir / "sample.pdf"
    pdf_file.write_text(SAMPLE_PDF_CONTENT)
    return pdf_file

@pytest.fixture
def sample_schema_file(temp_dir):
    """Create a temporary schema file for testing."""
    schema_file = temp_dir / "sample_schema.py"
    schema_file.write_text(SAMPLE_PYDANTIC_CODE)
    return schema_file

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for schema generation."""
    return SAMPLE_PYDANTIC_CODE

@pytest.fixture
def sample_patient_json(temp_dir):
    """Create a sample patient JSON file."""
    json_file = temp_dir / "patient.json"
    json_file.write_text(json.dumps(SAMPLE_PATIENT_DATA, indent=2))
    return json_file

@pytest.fixture
def mock_schema_registry(temp_dir):
    """Mock schema registry with sample data."""
    registry_file = temp_dir / "schema_registry.json"
    registry_data = {
        "test_schema_001": {
            "id": "test_schema_001",
            "class_name": "PatientRecord",
            "description": SAMPLE_SCHEMA_DESCRIPTION,
            "file_path": str(temp_dir / "test_schema.py"),
            "created_at": "2025-09-19T10:00:00",
            "model_used": "gpt-oss:120b",
            "temperature": 0.2
        }
    }
    registry_file.write_text(json.dumps(registry_data, indent=2))
    return registry_file

@pytest.fixture
def mock_console():
    """Mock rich console for testing CLI output."""
    with patch('mosaicx.display.console') as mock_console:
        yield mock_console

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing LLM interactions."""
    with patch('ollama.Client') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        yield mock_instance
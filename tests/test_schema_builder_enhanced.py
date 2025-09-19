"""
Test Schema Builder Module - Enhanced Test Cases

Comprehensive tests for the schema generation functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from mosaicx.schema_builder import synthesize_pydantic_model


class TestSchemaBuilder:
    """Test cases for schema builder functionality."""
    
    @patch('mosaicx.schema_builder.ollama.Client')
    def test_synthesize_pydantic_model_basic(self, mock_ollama):
        """Test basic schema synthesis functionality."""
        # Mock Ollama response
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        
        mock_response = {
            'message': {
                'content': '''```python
from pydantic import BaseModel, Field

class PatientRecord(BaseModel):
    """Patient demographic information."""
    name: str = Field(..., description="Patient full name")
    age: int = Field(..., description="Patient age")
```'''
            }
        }
        mock_client.chat.return_value = mock_response
        
        # Test schema generation
        result = synthesize_pydantic_model(
            description="Patient record with name and age",
            class_name="PatientRecord"
        )
        
        assert result is not None
        assert "class PatientRecord" in result
        assert "BaseModel" in result
        assert "name: str" in result
        assert "age: int" in result
    
    @patch('mosaicx.schema_builder.ollama.Client')
    def test_synthesize_with_custom_model(self, mock_ollama):
        """Test schema synthesis with custom model."""
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        
        mock_response = {
            'message': {
                'content': '''```python
class TestModel(BaseModel):
    test_field: str
```'''
            }
        }
        mock_client.chat.return_value = mock_response
        
        result = synthesize_pydantic_model(
            description="Simple test model",
            class_name="TestModel",
            model="mistral:latest"
        )
        
        # Verify model was used in the call
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]['model'] == 'mistral:latest'
    
    @patch('mosaicx.schema_builder.ollama.Client')
    def test_synthesize_with_temperature(self, mock_ollama):
        """Test schema synthesis with custom temperature."""
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        
        mock_response = {
            'message': {
                'content': 'class TestModel(BaseModel): pass'
            }
        }
        mock_client.chat.return_value = mock_response
        
        synthesize_pydantic_model(
            description="Test model",
            temperature=0.8
        )
        
        # Verify temperature was passed
        call_args = mock_client.chat.call_args
        assert call_args[1]['options']['temperature'] == 0.8
    
    @patch('mosaicx.schema_builder.ollama.Client')
    def test_synthesize_error_handling(self, mock_ollama):
        """Test error handling in schema synthesis."""
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        
        # Mock an exception
        mock_client.chat.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception):
            synthesize_pydantic_model(
                description="Test model"
            )
    
    @patch('mosaicx.schema_builder.ollama.Client')
    def test_synthesize_malformed_response(self, mock_ollama):
        """Test handling of malformed LLM response."""
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        
        # Mock response without proper Python code
        mock_response = {
            'message': {
                'content': 'This is not Python code'
            }
        }
        mock_client.chat.return_value = mock_response
        
        result = synthesize_pydantic_model(
            description="Test model"
        )
        
        # Should still return the content even if not proper Python
        assert result is not None
        assert isinstance(result, str)
    
    @patch('mosaicx.schema_builder.ollama.Client')
    def test_synthesize_complex_description(self, mock_ollama):
        """Test schema synthesis with complex medical description."""
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        
        mock_response = {
            'message': {
                'content': '''```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class MedicalRecord(BaseModel):
    """Comprehensive medical record."""
    patient_id: str = Field(..., description="Unique patient identifier")
    admission_date: datetime = Field(..., description="Date of admission")
    diagnoses: List[str] = Field(default_factory=list, description="List of diagnoses")
    medications: Optional[List[str]] = Field(None, description="Current medications")
    vital_signs: Optional[dict] = Field(None, description="Latest vital signs")
```'''
            }
        }
        mock_client.chat.return_value = mock_response
        
        complex_description = """
        Medical record containing patient identification, admission details,
        multiple diagnoses, medication list, and vital signs measurements
        including blood pressure, heart rate, and temperature readings.
        """
        
        result = synthesize_pydantic_model(
            description=complex_description,
            class_name="MedicalRecord"
        )
        
        assert "class MedicalRecord" in result
        assert "patient_id" in result
        assert "admission_date" in result
        assert "diagnoses" in result
        assert "medications" in result
        assert "vital_signs" in result
    
    def test_synthesize_invalid_inputs(self):
        """Test schema synthesis with invalid inputs."""
        # Test empty description
        with pytest.raises(ValueError):
            synthesize_pydantic_model(description="")
        
        # Test None description
        with pytest.raises(ValueError):
            synthesize_pydantic_model(description=None)
    
    @patch('mosaicx.schema_builder.ollama.Client')
    def test_synthesize_with_api_parameters(self, mock_ollama):
        """Test schema synthesis with custom API parameters."""
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        
        mock_response = {
            'message': {
                'content': 'class TestModel(BaseModel): pass'
            }
        }
        mock_client.chat.return_value = mock_response
        
        # Test with custom base_url and api_key
        synthesize_pydantic_model(
            description="Test model",
            base_url="http://custom-api.com",
            api_key="test-key-123"
        )
        
        # Verify the client was created with custom parameters
        mock_ollama.assert_called_once()
        # Note: The actual API parameter handling would need to be implemented
        # in the schema_builder module for this test to be meaningful
"""
Test Schema Builder Module

Tests for schema generation functionality including LLM integration,
Pydantic model synthesis, and code generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from mosaicx.schema.builder import synthesize_pydantic_model


class TestSchemaBuilder:
    """Test cases for schema generation functionality."""
    
    @patch('mosaicx.schema.builder.ollama.Client')
    def test_synthesize_pydantic_model_success(self, mock_client_class):
        """Test successful schema generation."""
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock successful LLM response
        mock_llm_response = '''```python
class PatientInfo(BaseModel):
    name: str
    age: int
```'''
        
        mock_client.chat.return_value = {
            'message': {'content': mock_llm_response}
        }
        
        # Test schema generation
        result = synthesize_pydantic_model(
            description="Patient demographics with name and age",
            class_name="PatientRecord",
            model="gpt-oss:120b"
        )
        
        # Assertions
        assert result is not None
        assert "class PatientRecord(BaseModel)" in result
        mock_client.chat.assert_called_once()
        
    @patch('mosaicx.schema.builder.get_ollama_client')
    def test_synthesize_pydantic_model_failure(self, mock_get_client):
        """Test schema generation failure handling."""
        # Setup mock client to raise exception
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.side_effect = Exception("Connection error")
        
        # Test that exception is raised
        with pytest.raises(Exception):
            synthesize_pydantic_model(
                description="Patient data",
                class_name="Patient",
                model="gpt-oss:120b"
            )
    
    @patch('mosaicx.schema.builder.get_ollama_client')
    def test_synthesize_with_custom_parameters(self, mock_get_client, mock_llm_response):
        """Test schema generation with custom parameters."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.return_value = {
            'message': {'content': mock_llm_response}
        }
        
        result = synthesize_pydantic_model(
            description="Complex medical record",
            class_name="MedicalRecord",
            model="mistral:latest",
            base_url="http://localhost:11434",
            api_key="test_key",
            temperature=0.5
        )
        
        assert result is not None
        assert "class MedicalRecord(BaseModel)" in result or "class PatientRecord(BaseModel)" in result
        
        # Check that chat was called with correct temperature
        call_args = mock_client.chat.call_args
        assert call_args[1]['options']['temperature'] == 0.5
    
    def test_invalid_model_name(self):
        """Test behavior with invalid model name."""
        with patch('mosaicx.schema.builder.get_ollama_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.chat.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception):
                synthesize_pydantic_model(
                    description="Test schema",
                    class_name="TestModel",
                    model="nonexistent:model"
                )


class TestPromptGeneration:
    """Test cases for prompt generation and LLM interaction."""
    
    def test_prompt_structure(self):
        """Test that generated prompts have correct structure."""
        with patch('mosaicx.schema.builder.get_ollama_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.chat.return_value = {
                'message': {'content': 'mock_response'}
            }
            
            synthesize_pydantic_model(
                description="Patient vital signs",
                class_name="VitalSigns",
                model="gpt-oss:120b"
            )
            
            # Get the prompt that was sent
            call_args = mock_client.chat.call_args
            messages = call_args[1]['messages']
            
            # Check prompt structure
            assert len(messages) > 0
            assert any('Patient vital signs' in str(msg) for msg in messages)
            assert any('VitalSigns' in str(msg) for msg in messages)
    
    def test_temperature_parameter(self):
        """Test that temperature parameter is correctly passed."""
        with patch('mosaicx.schema.builder.get_ollama_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.chat.return_value = {
                'message': {'content': 'mock_response'}
            }
            
            synthesize_pydantic_model(
                description="Test",
                class_name="Test",
                model="gpt-oss:120b",
                temperature=0.8
            )
            
            call_args = mock_client.chat.call_args
            assert call_args[1]['options']['temperature'] == 0.8
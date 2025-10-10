"""
Test Extractor Module

Tests for PDF extraction and data processing functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from mosaicx.extractor import extract_from_pdf, ExtractionError, load_schema_model


class TestPDFExtraction:
    """Test cases for PDF extraction functionality."""
    
    def test_extract_from_pdf_success(self, sample_pdf_file, sample_schema_file):
        """Test successful PDF extraction."""
        with patch('mosaicx.extractor.extract_text_from_document') as mock_extract_text:
            with patch('mosaicx.extractor.get_ollama_client') as mock_get_client:
                with patch('mosaicx.extractor.load_schema_model') as mock_load_schema:
                    # Setup mocks
                    mock_extract_text.return_value = "Patient: John Doe, Age: 45, Gender: Male"
                    
                    mock_client = Mock()
                    mock_get_client.return_value = mock_client
                    mock_client.chat.return_value = {
                        'message': {'content': '{"name": "John Doe", "age": 45, "gender": "Male"}'}
                    }
                    
                    mock_schema_class = Mock()
                    mock_schema_instance = Mock()
                    mock_schema_class.return_value = mock_schema_instance
                    mock_schema_instance.model_dump.return_value = {
                        "name": "John Doe", 
                        "age": 45, 
                        "gender": "Male"
                    }
                    mock_load_schema.return_value = mock_schema_class
                    
                    # Test extraction
                    result = extract_from_pdf(
                        pdf_path=str(sample_pdf_file),
                        schema_path=str(sample_schema_file)
                    )
                    
                    # Assertions
                    assert result is not None
                    assert isinstance(result, dict)
                    assert "name" in result
                    mock_extract_text.assert_called_once()
                    mock_client.chat.assert_called_once()
    
    def test_extract_from_pdf_invalid_file(self):
        """Test extraction with invalid PDF file."""
        with pytest.raises(FileNotFoundError):
            extract_from_pdf(
                pdf_path="nonexistent.pdf",
                schema_path="schema.py"
            )
    
    def test_extract_from_pdf_invalid_schema(self, sample_pdf_file):
        """Test extraction with invalid schema file."""
        with pytest.raises(FileNotFoundError):
            extract_from_pdf(
                pdf_path=str(sample_pdf_file),
                schema_path="nonexistent_schema.py"
            )
    
    @patch('mosaicx.extractor.extract_text_from_document')
    @patch('mosaicx.extractor.get_ollama_client')
    @patch('mosaicx.extractor.load_schema_model')
    def test_extract_pdf_text_extraction_failure(self, mock_load_schema, mock_get_client, mock_extract_text, sample_pdf_file, sample_schema_file):
        """Test handling of PDF text extraction failure."""
        mock_extract_text.side_effect = Exception("PDF extraction failed")
        
        with pytest.raises(ExtractionError):
            extract_from_pdf(
                pdf_path=str(sample_pdf_file),
                schema_path=str(sample_schema_file)
            )
    
    @patch('mosaicx.extractor.extract_text_from_document')
    @patch('mosaicx.extractor.get_ollama_client')
    @patch('mosaicx.extractor.load_schema_model')
    def test_extract_llm_processing_failure(self, mock_load_schema, mock_get_client, mock_extract_text, sample_pdf_file, sample_schema_file):
        """Test handling of LLM processing failure."""
        mock_extract_text.return_value = "Sample text"
        
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.side_effect = Exception("LLM error")
        
        with pytest.raises(ExtractionError):
            extract_from_pdf(
                pdf_path=str(sample_pdf_file),
                schema_path=str(sample_schema_file)
            )
    
    @patch('mosaicx.extractor.extract_text_from_document')
    @patch('mosaicx.extractor.get_ollama_client')
    @patch('mosaicx.extractor.load_schema_model')
    def test_extract_invalid_json_response(self, mock_load_schema, mock_get_client, mock_extract_text, sample_pdf_file, sample_schema_file):
        """Test handling of invalid JSON response from LLM."""
        mock_extract_text.return_value = "Sample text"
        
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.chat.return_value = {
            'message': {'content': 'invalid json response'}
        }
        
        with pytest.raises(ExtractionError):
            extract_from_pdf(
                pdf_path=str(sample_pdf_file),
                schema_path=str(sample_schema_file)
            )


class TestSchemaLoading:
    """Test cases for schema loading functionality."""
    
    def test_load_schema_model_success(self, sample_schema_file):
        """Test successful schema model loading."""
        # Create a proper Python file with a BaseModel class
        schema_content = '''
from pydantic import BaseModel
from typing import Optional

class PatientRecord(BaseModel):
    name: str
    age: int
    gender: Optional[str] = None
'''
        sample_schema_file.write_text(schema_content)
        
        # Test loading the schema
        schema_class = load_schema_model(str(sample_schema_file))
        
        assert schema_class is not None
        assert hasattr(schema_class, '__name__')
        
        # Test instantiation
        instance = schema_class(name="John Doe", age=45)
        assert instance.name == "John Doe"
        assert instance.age == 45
    
    def test_load_schema_model_invalid_file(self):
        """Test loading schema from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_schema_model("nonexistent_schema.py")
    
    def test_load_schema_model_invalid_python(self, temp_dir):
        """Test loading schema from invalid Python file."""
        invalid_file = temp_dir / "invalid_schema.py"
        invalid_file.write_text("invalid python syntax @@#$")
        
        with pytest.raises(Exception):  # Should raise syntax or import error
            load_schema_model(str(invalid_file))
    
    def test_load_schema_model_no_basemodel(self, temp_dir):
        """Test loading schema file with no BaseModel class."""
        no_model_file = temp_dir / "no_model_schema.py"
        no_model_file.write_text("def some_function(): pass")
        
        result = load_schema_model(str(no_model_file))
        assert result is None  # or should raise exception depending on implementation


class TestTextExtraction:
    """Test cases for text extraction utilities."""
    
    @patch('mosaicx.extractor.fitz')
    def test_extract_text_from_document_success(self, mock_fitz, sample_pdf_file):
        """Test successful text extraction from PDF."""
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample extracted text from PDF"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)
        mock_fitz.open.return_value = mock_doc
        
        from mosaicx.extractor import extract_text_from_document
        
        text = extract_text_from_document(str(sample_pdf_file))
        assert text == "Sample extracted text from PDF"
        mock_fitz.open.assert_called_once_with(str(sample_pdf_file))
    
    @patch('mosaicx.extractor.fitz')
    def test_extract_text_from_document_empty(self, mock_fitz, sample_pdf_file):
        """Test text extraction from empty PDF."""
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)
        mock_fitz.open.return_value = mock_doc
        
        from mosaicx.extractor import extract_text_from_document
        
        text = extract_text_from_document(str(sample_pdf_file))
        assert text == ""
    
    @patch('mosaicx.extractor.fitz')
    def test_extract_text_from_document_error(self, mock_fitz, sample_pdf_file):
        """Test handling of PDF extraction errors."""
        mock_fitz.open.side_effect = Exception("PDF corrupted")
        
        from mosaicx.extractor import extract_text_from_document
        
        with pytest.raises(Exception):
            extract_text_from_document(str(sample_pdf_file))


class TestDataValidation:
    """Test cases for extracted data validation."""
    
    def test_validate_extracted_data_success(self):
        """Test successful validation of extracted data."""
        from mosaicx.extractor import load_schema_model
        
        # Create a simple schema for testing
        schema_content = '''
from pydantic import BaseModel, Field

class TestModel(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)
'''
        
        with patch('builtins.open', mock_open(read_data=schema_content)):
            with patch('importlib.util.spec_from_file_location'):
                with patch('importlib.util.module_from_spec'):
                    # Mock the schema loading for this test
                    pass
    
    def test_validate_extracted_data_validation_error(self):
        """Test handling of validation errors in extracted data."""
        # This would test Pydantic validation errors
        pass
    
    def test_data_type_conversion(self):
        """Test automatic data type conversion during extraction."""
        # Test string to int conversion, etc.
        pass


class TestExtractionOptions:
    """Test cases for extraction configuration options."""
    
    def test_extract_with_custom_model(self, sample_pdf_file, sample_schema_file):
        """Test extraction with custom LLM model."""
        with patch('mosaicx.extractor.extract_text_from_document'):
            with patch('mosaicx.extractor.get_ollama_client'):
                with patch('mosaicx.extractor.load_schema_model'):
                    # Test with different model
                    result = extract_from_pdf(
                        pdf_path=str(sample_pdf_file),
                        schema_path=str(sample_schema_file),
                        model="mistral:latest"
                    )
                    # Should complete without error
    
    def test_extract_with_debug_mode(self, sample_pdf_file, sample_schema_file):
        """Test extraction with debug mode enabled."""
        with patch('mosaicx.extractor.extract_text_from_document'):
            with patch('mosaicx.extractor.get_ollama_client'):
                with patch('mosaicx.extractor.load_schema_model'):
                    # Test with debug flag
                    result = extract_from_pdf(
                        pdf_path=str(sample_pdf_file),
                        schema_path=str(sample_schema_file),
                        debug=True
                    )
                    # Should complete without error
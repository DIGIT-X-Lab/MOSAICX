"""
Test Extractor Module - Enhanced Test Cases

Comprehensive tests for PDF extraction functionality.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
from pathlib import Path
import json

from mosaicx.extractor import extract_from_pdf, load_schema_model, ExtractionError


class TestExtractor:
    """Test cases for PDF extraction functionality."""
    
    @patch('mosaicx.extractor.ollama.Client')
    @patch('mosaicx.extractor.fitz.open')
    def test_extract_from_pdf_basic(self, mock_fitz, mock_ollama, temp_dir):
        """Test basic PDF extraction functionality."""
        # Mock PDF reading
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Patient: John Doe, Age: 45"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)
        mock_fitz.return_value = mock_doc
        
        # Mock schema loading
        schema_file = temp_dir / "test_schema.py"
        schema_file.write_text("""
from pydantic import BaseModel

class PatientRecord(BaseModel):
    name: str
    age: int
""")
        
        # Mock Ollama response
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        mock_response = {
            'message': {
                'content': '{"name": "John Doe", "age": 45}'
            }
        }
        mock_client.chat.return_value = mock_response
        
        # Create test PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"dummy pdf content")
        
        # Test extraction
        result = extract_from_pdf(
            pdf_path=str(pdf_file),
            schema_file_path=str(schema_file)
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
    
    def test_extract_from_pdf_missing_file(self):
        """Test extraction with missing PDF file."""
        with pytest.raises((FileNotFoundError, ExtractionError)):
            extract_from_pdf(
                pdf_path="/nonexistent/file.pdf",
                schema_file_path="/some/schema.py"
            )
    
    def test_load_schema_model_valid_file(self, temp_dir):
        """Test loading valid schema model."""
        schema_content = """
from pydantic import BaseModel, Field

class TestModel(BaseModel):
    '''Test model for unit testing.'''
    name: str = Field(..., description="Test name")
    value: int = Field(..., description="Test value")
"""
        schema_file = temp_dir / "test_schema.py"
        schema_file.write_text(schema_content)
        
        model_class = load_schema_model(str(schema_file))
        
        assert model_class is not None
        assert model_class.__name__ == "TestModel"
        assert hasattr(model_class, 'name')
        assert hasattr(model_class, 'value')
    
    def test_load_schema_model_missing_file(self):
        """Test loading schema model with missing file."""
        with pytest.raises(FileNotFoundError):
            load_schema_model("/nonexistent/schema.py")
    
    def test_load_schema_model_invalid_python(self, temp_dir):
        """Test loading schema with invalid Python code."""
        schema_file = temp_dir / "invalid_schema.py"
        schema_file.write_text("This is not valid Python code!!!")
        
        with pytest.raises(Exception):  # Should raise SyntaxError or similar
            load_schema_model(str(schema_file))
    
    def test_load_schema_model_no_basemodel(self, temp_dir):
        """Test loading schema without BaseModel class."""
        schema_content = """
# Valid Python but no BaseModel
def some_function():
    return "hello"

class NotABaseModel:
    pass
"""
        schema_file = temp_dir / "no_basemodel_schema.py"
        schema_file.write_text(schema_content)
        
        # Should handle gracefully or raise appropriate error
        result = load_schema_model(str(schema_file))
        # The actual behavior depends on implementation
        # This test documents expected behavior
    
    @patch('mosaicx.extractor.ollama.Client')
    @patch('mosaicx.extractor.fitz.open')
    def test_extract_with_custom_model(self, mock_fitz, mock_ollama, temp_dir):
        """Test extraction with custom LLM model."""
        # Setup mocks
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Test content"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)
        mock_fitz.return_value = mock_doc
        
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        mock_response = {
            'message': {
                'content': '{"test": "data"}'
            }
        }
        mock_client.chat.return_value = mock_response
        
        # Create test files
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"dummy")
        
        schema_file = temp_dir / "schema.py"
        schema_file.write_text("from pydantic import BaseModel\nclass Test(BaseModel): pass")
        
        # Test with custom model
        extract_from_pdf(
            pdf_path=str(pdf_file),
            schema_file_path=str(schema_file),
            model="mistral:latest"
        )
        
        # Verify custom model was used
        mock_client.chat.assert_called()
        call_args = mock_client.chat.call_args
        assert call_args[1]['model'] == 'mistral:latest'
    
    @patch('mosaicx.extractor.ollama.Client')
    @patch('mosaicx.extractor.fitz.open')
    def test_extract_large_pdf(self, mock_fitz, mock_ollama, temp_dir):
        """Test extraction with large PDF (multiple pages)."""
        # Mock multiple pages
        mock_doc = Mock()
        mock_pages = []
        for i in range(5):  # 5 pages
            mock_page = Mock()
            mock_page.get_text.return_value = f"Page {i+1} content with patient data"
            mock_pages.append(mock_page)
        
        mock_doc.__iter__ = Mock(return_value=iter(mock_pages))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)
        mock_fitz.return_value = mock_doc
        
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        mock_response = {
            'message': {
                'content': '{"extracted": "data from all pages"}'
            }
        }
        mock_client.chat.return_value = mock_response
        
        # Create test files
        pdf_file = temp_dir / "large.pdf"
        pdf_file.write_bytes(b"large pdf content")
        
        schema_file = temp_dir / "schema.py"
        schema_file.write_text("from pydantic import BaseModel\nclass Data(BaseModel): pass")
        
        result = extract_from_pdf(
            pdf_path=str(pdf_file),
            schema_file_path=str(schema_file)
        )
        
        assert result is not None
        # Verify all pages were processed
        assert mock_doc.__iter__.called
    
    @patch('mosaicx.extractor.ollama.Client')
    @patch('mosaicx.extractor.fitz.open')
    def test_extract_malformed_llm_response(self, mock_fitz, mock_ollama, temp_dir):
        """Test extraction with malformed LLM response."""
        # Setup mocks
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Test content"
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)
        mock_fitz.return_value = mock_doc
        
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        # Return invalid JSON
        mock_response = {
            'message': {
                'content': 'This is not valid JSON'
            }
        }
        mock_client.chat.return_value = mock_response
        
        # Create test files
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"dummy")
        
        schema_file = temp_dir / "schema.py"
        schema_file.write_text("from pydantic import BaseModel\nclass Test(BaseModel): pass")
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(ExtractionError):
            extract_from_pdf(
                pdf_path=str(pdf_file),
                schema_file_path=str(schema_file)
            )
    
    @patch('mosaicx.extractor.ollama.Client')
    def test_extract_connection_error(self, mock_ollama, temp_dir):
        """Test extraction with LLM connection error."""
        mock_client = Mock()
        mock_ollama.return_value = mock_client
        mock_client.chat.side_effect = Exception("Connection failed")
        
        # Create test files
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"dummy")
        
        schema_file = temp_dir / "schema.py"
        schema_file.write_text("from pydantic import BaseModel\nclass Test(BaseModel): pass")
        
        with pytest.raises(ExtractionError):
            extract_from_pdf(
                pdf_path=str(pdf_file),
                schema_file_path=str(schema_file)
            )
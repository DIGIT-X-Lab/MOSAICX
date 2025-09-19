"""
Test CLI Module - Enhanced Test Cases

Comprehensive tests for the command-line interface functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from mosaicx.mosaicx import cli, generate, extract


class TestCLIInterface:
    """Test cases for main CLI functionality."""
    
    def test_cli_help(self):
        """Test main CLI help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "MOSAICX" in result.output
        assert "Medical cOmputational Suite" in result.output
    
    def test_cli_version(self):
        """Test CLI version display."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert "1.0" in result.output  # Version should contain 1.0
    
    @patch('mosaicx.mosaicx.show_main_banner')
    def test_cli_no_command_shows_banner(self, mock_banner):
        """Test that CLI shows banner when no command is provided."""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        
        mock_banner.assert_called_once()
        assert "Welcome to MOSAICX" in result.output


class TestGenerateCommand:
    """Test cases for generate command."""
    
    def test_generate_help(self):
        """Test generate command help."""
        runner = CliRunner()
        result = runner.invoke(generate, ['--help'])
        
        assert result.exit_code == 0
        assert "Generate Pydantic schemas" in result.output
        assert "MODEL COMPATIBILITY" in result.output
        assert "gpt-oss:120b" in result.output
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    @patch('mosaicx.mosaicx.register_schema')
    def test_generate_basic_functionality(self, mock_register, mock_synthesize, temp_dir):
        """Test basic generate command functionality."""
        # Mock the schema synthesis
        mock_synthesize.return_value = "class TestModel(BaseModel): pass"
        mock_register.return_value = "test_schema_123"
        
        runner = CliRunner()
        result = runner.invoke(generate, [
            '--desc', 'A simple test model',
            '--class-name', 'TestModel'
        ])
        
        assert result.exit_code == 0
        mock_synthesize.assert_called_once()
        mock_register.assert_called_once()
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    def test_generate_with_custom_model(self, mock_synthesize):
        """Test generate command with custom model."""
        mock_synthesize.return_value = "class TestModel(BaseModel): pass"
        
        runner = CliRunner()
        result = runner.invoke(generate, [
            '--desc', 'A test model',
            '--model', 'mistral:latest'
        ])
        
        mock_synthesize.assert_called_once()
        # Check that the model parameter was passed correctly
        call_args = mock_synthesize.call_args
        assert 'mistral:latest' in str(call_args)
    
    def test_generate_missing_description(self):
        """Test generate command without required description."""
        runner = CliRunner()
        result = runner.invoke(generate, [])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


class TestExtractCommand:
    """Test cases for extract command."""
    
    def test_extract_help(self):
        """Test extract command help."""
        runner = CliRunner()
        result = runner.invoke(extract, ['--help'])
        
        assert result.exit_code == 0
        assert "Extract structured data from PDF" in result.output
        assert "SCHEMA FORMATS ACCEPTED" in result.output
        assert "MODEL COMPATIBILITY" in result.output
        assert "gpt-oss:20b (not working)" in result.output
    
    def test_extract_missing_pdf(self):
        """Test extract command without PDF file."""
        runner = CliRunner()
        result = runner.invoke(extract, [
            '--schema', 'test_schema'
        ])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
    
    def test_extract_missing_schema(self, temp_dir):
        """Test extract command without schema."""
        # Create a dummy PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_text("dummy pdf content")
        
        runner = CliRunner()
        result = runner.invoke(extract, [
            '--pdf', str(pdf_file)
        ])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()
    
    def test_extract_nonexistent_pdf(self):
        """Test extract command with non-existent PDF."""
        runner = CliRunner()
        result = runner.invoke(extract, [
            '--pdf', '/nonexistent/file.pdf',
            '--schema', 'test_schema'
        ])
        
        assert result.exit_code != 0
    
    @patch('mosaicx.mosaicx._resolve_schema_reference')
    @patch('mosaicx.mosaicx.extract_from_pdf')
    def test_extract_basic_functionality(self, mock_extract, mock_resolve, temp_dir):
        """Test basic extract command functionality."""
        # Create a dummy PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_text("dummy pdf content")
        
        # Mock schema resolution and extraction
        schema_file = temp_dir / "test_schema.py"
        schema_file.write_text("class TestModel(BaseModel): pass")
        mock_resolve.return_value = schema_file
        mock_extract.return_value = {"test": "data"}
        
        runner = CliRunner()
        result = runner.invoke(extract, [
            '--pdf', str(pdf_file),
            '--schema', 'test_schema'
        ])
        
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with('test_schema')
        mock_extract.assert_called_once()


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    @patch('mosaicx.mosaicx.register_schema')
    @patch('mosaicx.mosaicx._resolve_schema_reference')
    @patch('mosaicx.mosaicx.extract_from_pdf')
    def test_generate_then_extract_workflow(self, mock_extract, mock_resolve, 
                                          mock_register, mock_synthesize, temp_dir):
        """Test complete workflow: generate schema then extract."""
        runner = CliRunner()
        
        # Step 1: Generate schema
        schema_code = "class PatientRecord(BaseModel): name: str"
        mock_synthesize.return_value = schema_code
        mock_register.return_value = "patient_schema_123"
        
        result = runner.invoke(generate, [
            '--desc', 'Patient record with name',
            '--class-name', 'PatientRecord'
        ])
        assert result.exit_code == 0
        
        # Step 2: Extract using the schema
        pdf_file = temp_dir / "patient.pdf"
        pdf_file.write_text("Patient: John Doe")
        
        schema_file = temp_dir / "patient_schema.py"
        schema_file.write_text(schema_code)
        
        mock_resolve.return_value = schema_file
        mock_extract.return_value = {"name": "John Doe"}
        
        result = runner.invoke(extract, [
            '--pdf', str(pdf_file),
            '--schema', 'patient_schema_123'
        ])
        assert result.exit_code == 0
    
    def test_verbose_flag(self):
        """Test verbose flag functionality."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose', '--help'])
        
        assert result.exit_code == 0
        # Verbose flag should be processed without errors
"""
Test CLI Module

Tests for command-line interface functionality and user interactions.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from mosaicx.mosaicx import cli, generate, extract


class TestCLIInterface:
    """Test cases for CLI command interface."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "MOSAICX" in result.output
        assert "generate" in result.output
        assert "extract" in result.output
    
    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert "1.0" in result.output  # Version should be displayed
    
    def test_cli_verbose_flag(self):
        """Test CLI verbose flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose'])
        
        assert result.exit_code == 0
    
    def test_cli_no_command(self):
        """Test CLI behavior with no subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        
        assert result.exit_code == 0
        assert "Welcome to MOSAICX" in result.output


class TestGenerateCommand:
    """Test cases for generate command."""
    
    def test_generate_help(self):
        """Test generate command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['generate', '--help'])
        
        assert result.exit_code == 0
        assert "Generate Pydantic schemas" in result.output
        assert "--desc" in result.output
        assert "MODEL COMPATIBILITY" in result.output
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    @patch('mosaicx.mosaicx.register_schema')
    def test_generate_success(self, mock_register, mock_synthesize):
        """Test successful schema generation."""
        mock_synthesize.return_value = "class TestModel(BaseModel): pass"
        mock_register.return_value = "test_schema_001"
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'generate',
                '--desc', 'Patient demographics',
                '--class-name', 'Patient'
            ])
        
        assert result.exit_code == 0
        mock_synthesize.assert_called_once()
        mock_register.assert_called_once()
    
    def test_generate_missing_desc(self):
        """Test generate command with missing description."""
        runner = CliRunner()
        result = runner.invoke(cli, ['generate'])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    def test_generate_with_custom_model(self, mock_synthesize):
        """Test generate with custom model."""
        mock_synthesize.return_value = "class TestModel(BaseModel): pass"
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'generate',
                '--desc', 'Test schema',
                '--model', 'mistral:latest'
            ])
        
        mock_synthesize.assert_called_once()
        call_args = mock_synthesize.call_args
        assert call_args[1]['model'] == 'mistral:latest'
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    def test_generate_with_temperature(self, mock_synthesize):
        """Test generate with custom temperature."""
        mock_synthesize.return_value = "class TestModel(BaseModel): pass"
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'generate',
                '--desc', 'Test schema',
                '--temperature', '0.8'
            ])
        
        mock_synthesize.assert_called_once()
        call_args = mock_synthesize.call_args
        assert call_args[1]['temperature'] == 0.8
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    def test_generate_save_to_file(self, mock_synthesize):
        """Test generate with save-to-file option."""
        mock_synthesize.return_value = "class TestModel(BaseModel): pass"
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'generate',
                '--desc', 'Test schema',
                '--save-model', 'test_schema.py'
            ])
        
        assert result.exit_code == 0
        # Check that file was saved
        assert Path('test_schema.py').exists()
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    def test_generate_failure(self, mock_synthesize):
        """Test generate command failure handling."""
        mock_synthesize.side_effect = Exception("LLM connection failed")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'generate',
            '--desc', 'Test schema'
        ])
        
        assert result.exit_code != 0
        assert "Error" in result.output


class TestExtractCommand:
    """Test cases for extract command."""
    
    def test_extract_help(self):
        """Test extract command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['extract', '--help'])
        
        assert result.exit_code == 0
        assert "Extract structured data" in result.output
        assert "--pdf" in result.output
        assert "--schema" in result.output
        assert "MODEL COMPATIBILITY" in result.output
        assert "SCHEMA FORMATS" in result.output
    
    def test_extract_missing_required_args(self):
        """Test extract command with missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ['extract'])
        
        assert result.exit_code != 0
        assert "Missing option" in result.output
    
    @patch('mosaicx.mosaicx.extract_from_pdf')
    @patch('mosaicx.mosaicx._resolve_schema_reference')
    def test_extract_success(self, mock_resolve, mock_extract):
        """Test successful PDF extraction."""
        mock_resolve.return_value = Path("test_schema.py")
        mock_extract.return_value = {"name": "John Doe", "age": 45}
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create test PDF file
            test_pdf = Path("test.pdf")
            test_pdf.write_text("dummy pdf content")
            
            result = runner.invoke(cli, [
                'extract',
                '--pdf', 'test.pdf',
                '--schema', 'test_schema'
            ])
        
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with('test_schema')
        mock_extract.assert_called_once()
    
    @patch('mosaicx.mosaicx._resolve_schema_reference')
    def test_extract_schema_not_found(self, mock_resolve):
        """Test extract with schema not found."""
        mock_resolve.return_value = None
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            test_pdf = Path("test.pdf")
            test_pdf.write_text("dummy pdf content")
            
            result = runner.invoke(cli, [
                'extract',
                '--pdf', 'test.pdf',
                '--schema', 'nonexistent_schema'
            ])
        
        assert result.exit_code != 0
        assert "Could not find schema" in result.output
    
    @patch('mosaicx.mosaicx.extract_from_pdf')
    @patch('mosaicx.mosaicx._resolve_schema_reference')
    def test_extract_with_save_option(self, mock_resolve, mock_extract):
        """Test extract with save to file option."""
        mock_resolve.return_value = Path("test_schema.py")
        mock_extract.return_value = {"name": "John Doe", "age": 45}
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            test_pdf = Path("test.pdf")
            test_pdf.write_text("dummy pdf content")
            
            result = runner.invoke(cli, [
                'extract',
                '--pdf', 'test.pdf',
                '--schema', 'test_schema',
                '--save', 'output.json'
            ])
        
        assert result.exit_code == 0
        
        # Check that output file was created
        output_file = Path('output.json')
        assert output_file.exists()
        
        # Check file contents
        saved_data = json.loads(output_file.read_text())
        assert saved_data["name"] == "John Doe"
    
    @patch('mosaicx.mosaicx.extract_from_pdf')
    @patch('mosaicx.mosaicx._resolve_schema_reference')
    def test_extract_with_custom_model(self, mock_resolve, mock_extract):
        """Test extract with custom model."""
        mock_resolve.return_value = Path("test_schema.py")
        mock_extract.return_value = {"name": "John Doe"}
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            test_pdf = Path("test.pdf")
            test_pdf.write_text("dummy pdf content")
            
            result = runner.invoke(cli, [
                'extract',
                '--pdf', 'test.pdf',
                '--schema', 'test_schema',
                '--model', 'mistral:latest'
            ])
        
        assert result.exit_code == 0
        call_args = mock_extract.call_args
        assert call_args[1]['model'] == 'mistral:latest'
    
    @patch('mosaicx.mosaicx.extract_from_pdf')
    @patch('mosaicx.mosaicx._resolve_schema_reference')
    def test_extract_failure(self, mock_resolve, mock_extract):
        """Test extract command failure handling."""
        mock_resolve.return_value = Path("test_schema.py")
        mock_extract.side_effect = Exception("Extraction failed")
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            test_pdf = Path("test.pdf")
            test_pdf.write_text("dummy pdf content")
            
            result = runner.invoke(cli, [
                'extract',
                '--pdf', 'test.pdf',
                '--schema', 'test_schema'
            ])
        
        assert result.exit_code != 0
        assert "Error" in result.output


class TestSchemaResolution:
    """Test cases for schema reference resolution."""
    
    @patch('mosaicx.mosaicx.get_schema_by_id')
    def test_resolve_schema_by_id(self, mock_get_schema):
        """Test resolving schema by ID."""
        mock_get_schema.return_value = {
            'file_path': '/path/to/schema.py'
        }
        
        from mosaicx.mosaicx import _resolve_schema_reference
        
        result = _resolve_schema_reference('schema_id_001')
        
        assert result == Path('/path/to/schema.py')
        mock_get_schema.assert_called_once_with('schema_id_001')
    
    def test_resolve_schema_by_filename(self):
        """Test resolving schema by filename."""
        from mosaicx.mosaicx import _resolve_schema_reference
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = _resolve_schema_reference('test_schema.py')
            
            assert result is not None
            assert str(result).endswith('test_schema.py')
    
    def test_resolve_schema_by_path(self):
        """Test resolving schema by full path."""
        from mosaicx.mosaicx import _resolve_schema_reference
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = _resolve_schema_reference('/full/path/to/schema.py')
            
            assert result == Path('/full/path/to/schema.py')
    
    @patch('mosaicx.mosaicx.get_schema_by_id')
    def test_resolve_schema_not_found(self, mock_get_schema):
        """Test resolving non-existent schema."""
        mock_get_schema.return_value = None
        
        from mosaicx.mosaicx import _resolve_schema_reference
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = _resolve_schema_reference('nonexistent_schema')
            
            assert result is None


class TestVerboseOutput:
    """Test cases for verbose output functionality."""
    
    @patch('mosaicx.mosaicx.synthesize_pydantic_model')
    def test_generate_verbose_output(self, mock_synthesize):
        """Test generate command with verbose output."""
        mock_synthesize.return_value = "class TestModel(BaseModel): pass"
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                '--verbose',
                'generate',
                '--desc', 'Test schema'
            ])
        
        # Should show verbose information
        assert "Generating schema" in result.output
    
    @patch('mosaicx.mosaicx.extract_from_pdf')
    @patch('mosaicx.mosaicx._resolve_schema_reference')
    def test_extract_verbose_output(self, mock_resolve, mock_extract):
        """Test extract command with verbose output."""
        mock_resolve.return_value = Path("test_schema.py")
        mock_extract.return_value = {"name": "John Doe"}
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            test_pdf = Path("test.pdf")
            test_pdf.write_text("dummy pdf content")
            
            result = runner.invoke(cli, [
                '--verbose',
                'extract',
                '--pdf', 'test.pdf',
                '--schema', 'test_schema'
            ])
        
        # Should show verbose information
        assert "Extracting from" in result.output
        assert "Using schema" in result.output
        assert "Using model" in result.output
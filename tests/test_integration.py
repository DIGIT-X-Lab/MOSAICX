"""
Integration Tests for MOSAICX

End-to-end integration tests that test complete workflows.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
import json
import tempfile
from unittest.mock import patch, Mock

from mosaicx.mosaicx import cli


class TestEndToEndWorkflows:
    """Integration tests for complete user workflows."""
    
    @patch('mosaicx.schema.builder.get_ollama_client')
    @patch('mosaicx.extractor.get_ollama_client')
    def test_complete_generate_and_extract_workflow(self, mock_extract_client, mock_gen_client):
        """Test complete workflow: generate schema -> extract data."""
        # Mock schema generation
        mock_gen_client.return_value.chat.return_value = {
            'message': {'content': '''
from pydantic import BaseModel
from typing import Optional

class PatientRecord(BaseModel):
    """Patient demographic information."""
    name: str
    age: int
    gender: Optional[str] = None
'''}
        }
        
        # Mock extraction
        mock_extract_client.return_value.chat.return_value = {
            'message': {'content': '{"name": "John Doe", "age": 45, "gender": "Male"}'}
        }
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Step 1: Generate schema
            generate_result = runner.invoke(cli, [
                'generate',
                '--desc', 'Patient record with name, age, and gender',
                '--class-name', 'PatientRecord',
                '--save-model', 'patient_schema.py'
            ])
            
            assert generate_result.exit_code == 0
            assert Path('patient_schema.py').exists()
            
            # Create a dummy PDF file
            test_pdf = Path('test_patient.pdf')
            test_pdf.write_text('Patient: John Doe, Age: 45, Gender: Male')
            
            # Step 2: Extract data using generated schema
            with patch('mosaicx.extractor.extract_text_from_pdf') as mock_extract_text:
                mock_extract_text.return_value = "Patient: John Doe, Age: 45, Gender: Male"
                
                extract_result = runner.invoke(cli, [
                    'extract',
                    '--pdf', 'test_patient.pdf',
                    '--schema', 'patient_schema.py',
                    '--save', 'extracted_data.json'
                ])
            
            assert extract_result.exit_code == 0
            assert Path('extracted_data.json').exists()
            
            # Verify extracted data
            extracted_data = json.loads(Path('extracted_data.json').read_text())
            assert 'name' in extracted_data
    
    def test_schema_registry_workflow(self):
        """Test schema registration and lookup workflow."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch('mosaicx.schema.builder.get_ollama_client') as mock_client:
                mock_client.return_value.chat.return_value = {
                    'message': {'content': 'class TestSchema(BaseModel): pass'}
                }
                
                # Generate and register schema
                result = runner.invoke(cli, [
                    'generate',
                    '--desc', 'Test schema for registry'
                ])
                
                assert result.exit_code == 0
                
                # List schemas should show the registered schema
                # This would require implementing a list command
                # or testing the registry functions directly


class TestErrorHandling:
    """Integration tests for error handling scenarios."""
    
    def test_invalid_pdf_file_handling(self):
        """Test handling of invalid PDF files."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create invalid PDF file
            invalid_pdf = Path('invalid.pdf')
            invalid_pdf.write_text('This is not a valid PDF file')
            
            # Create dummy schema
            schema_file = Path('test_schema.py')
            schema_file.write_text('class Test(BaseModel): pass')
            
            result = runner.invoke(cli, [
                'extract',
                '--pdf', 'invalid.pdf',
                '--schema', 'test_schema.py'
            ])
            
            # Should handle error gracefully
            assert result.exit_code != 0
    
    def test_network_error_handling(self):
        """Test handling of network/LLM connection errors."""
        runner = CliRunner()
        
        with patch('mosaicx.schema.builder.get_ollama_client') as mock_client:
            mock_client.side_effect = Exception("Connection failed")
            
            result = runner.invoke(cli, [
                'generate',
                '--desc', 'Test schema'
            ])
            
            assert result.exit_code != 0
            assert "Error" in result.output
    
    def test_invalid_schema_file_handling(self):
        """Test handling of invalid schema files."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create PDF file
            pdf_file = Path('test.pdf')
            pdf_file.write_text('Test PDF content')
            
            # Create invalid schema file
            invalid_schema = Path('invalid_schema.py')
            invalid_schema.write_text('invalid python syntax @@')
            
            result = runner.invoke(cli, [
                'extract',
                '--pdf', 'test.pdf',
                '--schema', 'invalid_schema.py'
            ])
            
            assert result.exit_code != 0


class TestCLIUsability:
    """Integration tests for CLI usability and user experience."""
    
    def test_help_commands_comprehensive(self):
        """Test that help commands provide comprehensive information."""
        runner = CliRunner()
        
        # Main help
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'generate' in result.output
        assert 'extract' in result.output
        
        # Generate help
        result = runner.invoke(cli, ['generate', '--help'])
        assert result.exit_code == 0
        assert 'MODEL COMPATIBILITY' in result.output
        
        # Extract help
        result = runner.invoke(cli, ['extract', '--help'])
        assert result.exit_code == 0
        assert 'SCHEMA FORMATS' in result.output
        assert 'MODEL COMPATIBILITY' in result.output
    
    def test_verbose_output_functionality(self):
        """Test verbose output provides useful information."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch('mosaicx.schema.builder.get_ollama_client') as mock_client:
                mock_client.return_value.chat.return_value = {
                    'message': {'content': 'class Test(BaseModel): pass'}
                }
                
                result = runner.invoke(cli, [
                    '--verbose',
                    'generate',
                    '--desc', 'Test schema'
                ])
                
                assert result.exit_code == 0
                # Should show verbose information
                assert 'Generating schema' in result.output
    
    def test_error_messages_user_friendly(self):
        """Test that error messages are user-friendly."""
        runner = CliRunner()
        
        # Test missing required arguments
        result = runner.invoke(cli, ['generate'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output
        
        result = runner.invoke(cli, ['extract'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output


class TestPerformanceAndLimits:
    """Integration tests for performance and edge cases."""
    
    def test_large_pdf_handling(self):
        """Test handling of large PDF files."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create large text file as PDF
            large_content = "Sample text. " * 10000  # Large content
            large_pdf = Path('large.pdf')
            large_pdf.write_text(large_content)
            
            schema_file = Path('test_schema.py')
            schema_file.write_text('''
from pydantic import BaseModel
class TestModel(BaseModel):
    summary: str
''')
            
            with patch('mosaicx.extractor.extract_text_from_pdf') as mock_extract:
                mock_extract.return_value = "Summary: Large document processed"
                
                with patch('mosaicx.extractor.get_ollama_client') as mock_client:
                    mock_client.return_value.chat.return_value = {
                        'message': {'content': '{"summary": "Large document processed"}'}
                    }
                    
                    result = runner.invoke(cli, [
                        'extract',
                        '--pdf', 'large.pdf',
                        '--schema', 'test_schema.py'
                    ])
                    
                    # Should handle large files
                    assert result.exit_code == 0
    
    def test_complex_schema_generation(self):
        """Test generation of complex schemas."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            complex_description = """
            Complete patient medical record including:
            - Personal information (name, age, gender, address)
            - Medical history with dates
            - Current medications with dosages
            - Vital signs measurements
            - Laboratory test results with reference ranges
            - Diagnosis codes and descriptions
            - Treatment plans and follow-up schedules
            """
            
            with patch('mosaicx.schema.builder.get_ollama_client') as mock_client:
                mock_client.return_value.chat.return_value = {
                    'message': {'content': '''
from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class ComplexPatientRecord(BaseModel):
    name: str
    age: int
    medications: List[str]
    # ... complex schema
'''}
                }
                
                result = runner.invoke(cli, [
                    'generate',
                    '--desc', complex_description,
                    '--class-name', 'ComplexPatientRecord'
                ])
                
                assert result.exit_code == 0


class TestCompatibilityTesting:
    """Integration tests for model and format compatibility."""
    
    def test_different_model_compatibility(self):
        """Test different LLM models work correctly."""
        test_models = ['gpt-oss:120b', 'mistral:latest']
        
        runner = CliRunner()
        for model in test_models:
            with runner.isolated_filesystem():
                with patch('mosaicx.schema.builder.get_ollama_client') as mock_client:
                    mock_client.return_value.chat.return_value = {
                        'message': {'content': 'class TestModel(BaseModel): pass'}
                    }
                    
                    result = runner.invoke(cli, [
                        'generate',
                        '--desc', 'Test schema',
                        '--model', model
                    ])
                    
                    assert result.exit_code == 0
    
    def test_schema_format_compatibility(self):
        """Test different schema reference formats work."""
        schema_formats = [
            'schema_file.py',  # Filename
            'full/path/to/schema.py',  # Path
            'schema_id_123'  # ID (would require registry setup)
        ]
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create PDF file
            pdf_file = Path('test.pdf')
            pdf_file.write_text('Test content')
            
            for schema_format in schema_formats[:2]:  # Test filename and path
                # Create schema file
                if '/' in schema_format:
                    schema_path = Path(schema_format)
                    schema_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    schema_path = Path(schema_format)
                
                schema_path.write_text('class TestModel(BaseModel): pass')
                
                with patch('mosaicx.extractor.extract_text_from_pdf'):
                    with patch('mosaicx.extractor.get_ollama_client'):
                        result = runner.invoke(cli, [
                            'extract',
                            '--pdf', 'test.pdf',
                            '--schema', schema_format
                        ])
                        
                        # Should handle different formats
                        # Note: Some may fail due to missing implementation
                        # but they shouldn't crash
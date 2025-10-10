"""
Real-World Integration Tests for MOSAICX

Tests using actual PDF data from the test datasets to validate end-to-end functionality.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
import json
import tempfile
from unittest.mock import patch, Mock

from mosaicx.mosaicx import cli
from mosaicx.extractor import extract_text_from_document


class TestRealWorldIntegration:
    """Integration tests using real PDF data."""
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Path to the real sample PDF file."""
        return Path(__file__).parent / "datasets" / "sample_patient_vitals.pdf"
    
    def test_pdf_text_extraction(self, sample_pdf_path):
        """Test that we can extract text from the real PDF file."""
        assert sample_pdf_path.exists(), f"Sample PDF not found at {sample_pdf_path}"
        
        # Extract text from the real PDF
        extracted_text = extract_text_from_document(str(sample_pdf_path))
        
        # Verify key information is extracted
        assert "Sarah Johnson" in extracted_text
        assert "PID-12345" in extracted_text
        assert "Blood Pressure" in extracted_text
        assert "128/82 mmHg" in extracted_text
        assert "Dr. Michael Chen" in extracted_text
        assert len(extracted_text) > 500  # Ensure substantial text extracted
    
    @patch('mosaicx.schema.builder.OpenAI')
    def test_schema_generation_for_patient_vitals(self, mock_openai, sample_pdf_path):
        """Test generating a schema suitable for patient vitals data."""
        # Mock the LLM response with a realistic patient vitals schema
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PatientVitals(BaseModel):
    """Patient vital signs and basic information."""
    patient_name: str = Field(..., description="Full name of the patient")
    patient_id: str = Field(..., description="Patient ID")
    date_of_birth: str = Field(..., description="Patient date of birth")
    age: int = Field(..., description="Patient age in years")
    gender: str = Field(..., description="Patient gender")
    visit_date: str = Field(..., description="Date of visit")
    blood_pressure: str = Field(..., description="Blood pressure reading")
    heart_rate: str = Field(..., description="Heart rate in bpm")
    temperature: str = Field(..., description="Body temperature")
    respiratory_rate: str = Field(..., description="Respiratory rate")
    oxygen_saturation: str = Field(..., description="Oxygen saturation percentage")
    weight: str = Field(..., description="Patient weight")
    height: str = Field(..., description="Patient height")
    bmi: str = Field(..., description="Body Mass Index")
    clinical_notes: Optional[str] = Field(None, description="Clinical notes from physician")
    attending_physician: Optional[str] = Field(None, description="Name of attending physician")
'''
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Generate schema for patient vitals
            result = runner.invoke(cli, [
                'generate',
                '--desc', 'Patient vital signs record with demographics and measurements',
                '--class-name', 'PatientVitals',
                '--save-model', 'patient_vitals_schema.py'
            ])
            
            assert result.exit_code == 0
            assert Path('patient_vitals_schema.py').exists()
            
            # Verify the generated schema file contains expected content
            schema_content = Path('patient_vitals_schema.py').read_text()
            assert 'PatientVitals' in schema_content
            assert 'patient_name' in schema_content
            assert 'blood_pressure' in schema_content

    @patch('mosaicx.schema.builder.OpenAI')
    @patch('mosaicx.extractor.get_ollama_client')
    def test_complete_extraction_workflow_with_real_pdf(
        self, mock_extract_client, mock_openai, sample_pdf_path
    ):
        """Test complete workflow using the real PDF file."""
        # Mock schema generation
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
from pydantic import BaseModel, Field
from typing import Optional

class PatientVitals(BaseModel):
    """Patient vital signs and basic information."""
    patient_name: str = Field(..., description="Full name of the patient")
    patient_id: str = Field(..., description="Patient ID")
    age: int = Field(..., description="Patient age in years")
    gender: str = Field(..., description="Patient gender")
    blood_pressure: str = Field(..., description="Blood pressure reading")
    heart_rate: str = Field(..., description="Heart rate in bpm")
    temperature: str = Field(..., description="Body temperature")
    clinical_notes: Optional[str] = Field(None, description="Clinical notes")
    attending_physician: Optional[str] = Field(None, description="Attending physician")
'''
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Mock extraction with realistic data from the PDF
        mock_extract_client.return_value.chat.return_value = {
            'message': {'content': json.dumps({
                "patient_name": "Sarah Johnson",
                "patient_id": "PID-12345",
                "age": 40,
                "gender": "Female",
                "blood_pressure": "128/82 mmHg",
                "heart_rate": "78 bpm",
                "temperature": "98.6°F (37.0°C)",
                "clinical_notes": "Patient presents with slightly elevated blood pressure. All other vital signs are within normal limits.",
                "attending_physician": "Dr. Michael Chen, MD"
            })}
        }
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Copy the real PDF to the test directory
            import shutil
            test_pdf = Path("sample_patient_vitals.pdf")
            shutil.copy(sample_pdf_path, test_pdf)
            
            # Step 1: Generate schema
            generate_result = runner.invoke(cli, [
                'generate',
                '--desc', 'Patient vital signs with demographics',
                '--class-name', 'PatientVitals',
                '--save-model', 'vitals_schema.py'
            ])
            
            assert generate_result.exit_code == 0
            assert Path('vitals_schema.py').exists()
            
            # Step 2: Extract data from real PDF
            extract_result = runner.invoke(cli, [
                'extract',
                '--document', str(test_pdf),
                '--schema', 'vitals_schema.py',
                '--save', 'extracted_vitals.json'
            ])
            
            assert extract_result.exit_code == 0
            assert Path('extracted_vitals.json').exists()
            
            # Step 3: Verify extracted data
            extracted_data = json.loads(Path('extracted_vitals.json').read_text())
            assert extracted_data['patient_name'] == "Sarah Johnson"
            assert extracted_data['patient_id'] == "PID-12345"
            assert extracted_data['age'] == 40
            assert extracted_data['gender'] == "Female"
            assert "128/82" in extracted_data['blood_pressure']
            assert "Dr. Michael Chen" in extracted_data['attending_physician']

    def test_schema_registry_with_real_schema(self, sample_pdf_path):
        """Test schema registry functionality with a realistic schema."""
        from mosaicx.schema.registry import SchemaRegistry, register_schema
        
        registry = SchemaRegistry()
        
        # Create a temporary schema file for patient vitals
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from pydantic import BaseModel, Field
from typing import Optional

class PatientVitals(BaseModel):
    """Patient vital signs and basic information."""
    patient_name: str = Field(..., description="Full name of the patient")
    patient_id: str = Field(..., description="Patient ID")
    blood_pressure: str = Field(..., description="Blood pressure reading")
    heart_rate: str = Field(..., description="Heart rate in bpm")
    attending_physician: Optional[str] = Field(None, description="Attending physician")
''')
            schema_path = f.name
        
        try:
            # Register the schema
            schema_info = register_schema(
                schema_path=schema_path,
                description="Patient vital signs schema",
                keywords=["medical", "vitals", "patient"]
            )
            
            assert schema_info is not None
            assert schema_info['class_name'] == 'PatientVitals'
            assert 'Patient vital signs' in schema_info['description']
            
            # Test registry lookup
            schemas = registry.list_schemas()
            assert len(schemas) > 0
            
            # Find our registered schema
            vital_schemas = [s for s in schemas if s.get('class_name') == 'PatientVitals']
            assert len(vital_schemas) > 0
            
        finally:
            # Cleanup
            Path(schema_path).unlink(missing_ok=True)

    def test_error_handling_with_invalid_pdf(self):
        """Test error handling when PDF processing fails."""
        from mosaicx.extractor import extract_text_from_document, ExtractionError
        
        # Test with non-existent file
        with pytest.raises((FileNotFoundError, ExtractionError)):
            extract_text_from_document("nonexistent.pdf")
        
        # Test with invalid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write("This is not a PDF file")
            invalid_pdf = f.name
        
        try:
            with pytest.raises(ExtractionError):
                extract_text_from_document(invalid_pdf)
        finally:
            Path(invalid_pdf).unlink(missing_ok=True)

    @patch('mosaicx.extractor.get_ollama_client')
    def test_extraction_with_malformed_llm_response(self, mock_client, sample_pdf_path):
        """Test handling of malformed LLM responses during extraction."""
        # Mock malformed JSON response
        mock_client.return_value.chat.return_value = {
            'message': {'content': 'This is not valid JSON'}
        }
        
        from mosaicx.extractor import extract_from_pdf
        
        # Create a simple schema file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from pydantic import BaseModel

class SimpleSchema(BaseModel):
    name: str
''')
            schema_path = f.name
        
        try:
            # This should handle the malformed response gracefully
            result = extract_from_pdf(str(sample_pdf_path), schema_path)
            # The function should either return None or raise a specific exception
            assert result is None or isinstance(result, dict)
        except Exception as e:
            # Should be a well-defined exception, not a generic parsing error
            assert "JSON" in str(e) or "parse" in str(e).lower()
        finally:
            Path(schema_path).unlink(missing_ok=True)


class TestPerformanceWithRealData:
    """Performance tests using real data."""
    
    def test_pdf_processing_performance(self, sample_pdf_path):
        """Test that PDF processing completes in reasonable time."""
        import time
        
        start_time = time.time()
        extracted_text = extract_text_from_document(str(sample_pdf_path))
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Should process in under 5 seconds
        assert len(extracted_text) > 0
        
    def test_schema_validation_performance(self):
        """Test schema validation performance with realistic data."""
        from pydantic import BaseModel, Field
        import time
        
        class PatientVitals(BaseModel):
            patient_name: str = Field(..., description="Full name of the patient")
            patient_id: str = Field(..., description="Patient ID")
            age: int = Field(..., description="Patient age in years")
            blood_pressure: str = Field(..., description="Blood pressure reading")
        
        # Test data based on our real PDF
        test_data = {
            "patient_name": "Sarah Johnson",
            "patient_id": "PID-12345",
            "age": 40,
            "blood_pressure": "128/82 mmHg"
        }
        
        start_time = time.time()
        
        # Validate 100 times to test performance
        for _ in range(100):
            validated_data = PatientVitals(**test_data)
            assert validated_data.patient_name == "Sarah Johnson"
        
        end_time = time.time()
        validation_time = end_time - start_time
        
        assert validation_time < 1.0  # Should complete 100 validations in under 1 second
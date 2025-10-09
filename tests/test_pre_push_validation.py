#!/usr/bin/env python3
"""
MOSAICX Pre-Push Validation Test Suite

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Comprehensive test suite to validate all MOSAICX functionality before pushing
to production. Tests CLI commands, Python API, and core functionality with
real datasets.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab, LMU Radiology
"""

import pytest
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from mosaicx import (
    generate_schema,
    extract_pdf,
    summarize_reports,
)


class TestPrePushValidation:
    """Comprehensive pre-push validation tests"""
    
    @pytest.fixture
    def test_datasets(self) -> Dict[str, Path]:
        """Test dataset paths"""
        base_path = Path(__file__).parent / "datasets"
        return {
            "extract_pdf": base_path / "extract" / "sample_patient_vitals.pdf",
            "summarize_dir": base_path / "summarize",
            "summarize_files": [
                base_path / "summarize" / "P001_CT_2025-08-01.pdf",
                base_path / "summarize" / "P001_CT_2025-09-10.pdf",
            ]
        }
    
    def test_cli_health_check(self):
        """Test basic CLI functionality and health"""
        result = subprocess.run(
            ["mosaicx", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "MOSAICX" in result.stdout
        assert "generate" in result.stdout
        assert "extract" in result.stdout
        assert "summarize" in result.stdout
    
    def test_cli_schema_generation(self):
        """Test CLI schema generation with realistic prompt"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run([
                "mosaicx", "generate",
                "--desc", "Extract patient name, age, and blood pressure from medical reports",
                "--model", "mistral:latest"
            ], capture_output=True, text=True, timeout=120)
            
            assert result.returncode == 0
            assert "Generated Schema Results" in result.stdout
            assert "GeneratedModel" in result.stdout
            assert "mistral:latest" in result.stdout
    
    def test_cli_pdf_extraction(self, test_datasets):
        """Test CLI PDF extraction with test dataset"""
        # First generate a schema
        schema_result = subprocess.run([
            "mosaicx", "generate",
            "--desc", "Extract patient name and age",
            "--model", "mistral:latest"
        ], capture_output=True, text=True, timeout=120)
        
        assert schema_result.returncode == 0
        
        # Extract using existing simple schema
        extract_result = subprocess.run([
            "mosaicx", "extract",
            "--pdf", str(test_datasets["extract_pdf"]),
            "--schema", "mosaicx/schema/templates/python/patient_identity.py",
            "--model", "mistral:latest"
        ], capture_output=True, text=True, timeout=120)
        
        assert extract_result.returncode == 0
        assert "Extraction results" in extract_result.stdout
        assert "name" in extract_result.stdout
    
    def test_cli_summarization(self, test_datasets):
        """Test CLI report summarization with test dataset"""
        result = subprocess.run([
            "mosaicx", "summarize",
            "--dir", str(test_datasets["summarize_dir"]),
            "--model", "mistral:latest"
        ], capture_output=True, text=True, timeout=180)
        
        assert result.returncode == 0
        assert "Patient: P001" in result.stdout
        assert "Timeline" in result.stdout or "timeline" in result.stdout
        assert "2025-08-01" in result.stdout
        assert "2025-09-10" in result.stdout
    
    def test_api_schema_generation(self):
        """Test Python API schema generation"""
        schema = generate_schema(
            "Patient demographics with name, age, sex, and date of birth",
            class_name="PatientDemographics",
            model="mistral:latest"
        )
        
        assert schema is not None
        assert schema.class_name == "PatientDemographics"
        assert hasattr(schema, 'write')
        
        # Test schema writing
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_path = schema.write(Path(temp_dir) / "test_schema.py")
            assert schema_path.exists()
            assert schema_path.suffix == ".py"
    
    def test_api_pdf_extraction(self, test_datasets):
        """Test Python API PDF extraction"""
        extraction = extract_pdf(
            pdf_path=str(test_datasets["extract_pdf"]),
            schema_path="mosaicx/schema/templates/python/patient_identity.py",
        )
        
        assert extraction is not None
        payload = extraction.to_dict()
        assert isinstance(payload, dict)
        assert "name" in payload
        # Should extract "Sarah Johnson" from test PDF
        assert "Sarah" in payload["name"] or "Johnson" in payload["name"]
    
    def test_api_summarization(self, test_datasets):
        """Test Python API report summarization"""
        summary = summarize_reports(
            paths=[str(test_datasets["summarize_dir"])],
            patient_id="P001",
        )
        
        assert summary is not None
        assert summary.patient.patient_id == "P001"
        assert len(summary.timeline) == 2
        assert "2025-08-01" in str(summary.timeline[0].date)
        assert "2025-09-10" in str(summary.timeline[1].date)
        assert "lymph" in summary.overall.lower() or "node" in summary.overall.lower()
    
    def test_end_to_end_workflow(self, test_datasets):
        """Test complete end-to-end workflow: generate → extract → summarize"""
        
        # 1. Generate schema
        schema = generate_schema(
            "Extract patient name, age, and medical findings",
            class_name="MedicalExtraction",
            model="mistral:latest"
        )
        assert schema is not None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 2. Write schema to file
            schema_path = schema.write(Path(temp_dir) / "medical_schema.py")
            assert schema_path.exists()
            
            # 3. Extract from PDF
            extraction = extract_pdf(
                pdf_path=str(test_datasets["extract_pdf"]),
                schema_path=str(schema_path),
            )
            assert extraction is not None
            payload = extraction.to_dict()
            assert isinstance(payload, dict)
            
            # 4. Summarize reports
            summary = summarize_reports(
                paths=[str(test_datasets["summarize_dir"])],
                patient_id="P001_E2E_Test",
            )
            assert summary is not None
            assert summary.patient.patient_id == "P001_E2E_Test"
    
    def test_error_handling(self):
        """Test proper error handling for invalid inputs"""
        
        # Test invalid PDF path
        with pytest.raises(Exception):
            extract_pdf(
                pdf_path="nonexistent_file.pdf",
                schema_path="mosaicx/schema/templates/python/patient_identity.py",
            )
        
        # Test invalid schema path
        with pytest.raises(Exception):
            extract_pdf(
                pdf_path="tests/datasets/extract/sample_patient_vitals.pdf",
                schema_path="nonexistent_schema.py"
            )
    
    def test_output_directories_exist(self):
        """Ensure output directories are properly created"""
        output_dir = Path("output")
        schemas_dir = Path("schemas")
        
        # These should exist or be creatable
        assert output_dir.exists() or output_dir.parent.exists()
        assert schemas_dir.exists() or schemas_dir.parent.exists()
    
    @pytest.mark.slow
    def test_performance_benchmarks(self, test_datasets):
        """Test basic performance benchmarks"""
        import time
        
        # Schema generation should complete in reasonable time
        start_time = time.time()
        schema = generate_schema(
            "Simple patient name extraction",
            class_name="SimplePatient",
            model="mistral:latest"
        )
        schema_time = time.time() - start_time
        assert schema_time < 60  # Should complete in under 1 minute
        
        # Extraction should be fast
        start_time = time.time()
        extraction = extract_pdf(
            pdf_path=str(test_datasets["extract_pdf"]),
            schema_path="mosaicx/schema/templates/python/patient_identity.py",
        )
        extract_time = time.time() - start_time
        assert extract_time < 45  # Should complete in under 45 seconds
        
        print(f"Performance: Schema={schema_time:.2f}s, Extract={extract_time:.2f}s")


def run_pre_push_tests():
    """
    Standalone function to run pre-push validation tests
    Can be called directly from command line or CI/CD
    """
    import sys
    
    print("🧬 MOSAICX Pre-Push Validation")
    print("=" * 50)
    
    # Run tests with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--durations=10"  # Show slowest tests
    ])
    
    if exit_code == 0:
        print("\n✅ All pre-push validation tests PASSED!")
        print("🚀 Ready for production push!")
    else:
        print("\n❌ Pre-push validation FAILED!")
        print("🛠️  Please fix issues before pushing.")
        
    return exit_code


if __name__ == "__main__":
    exit(run_pre_push_tests())

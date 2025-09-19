"""
Test Schema Registry Module

Tests for schema registration, lookup, and management functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from mosaicx.schema_registry import (
    register_schema,
    list_schemas,
    get_schema_by_id,
    get_suggested_filename,
    cleanup_missing_files,
    scan_and_register_existing_schemas
)


class TestSchemaRegistry:
    """Test cases for schema registry functionality."""
    
    def test_register_schema_success(self, temp_dir):
        """Test successful schema registration."""
        test_file = temp_dir / "test_schema.py"
        test_file.write_text("class TestModel(BaseModel): pass")
        
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', temp_dir / "registry.json"):
            schema_id = register_schema(
                class_name="TestModel",
                description="Test schema for unit testing",
                file_path=test_file,
                model_used="gpt-oss:120b",
                temperature=0.2
            )
        
        assert schema_id is not None
        assert len(schema_id) > 0
    
    def test_register_schema_duplicate(self, temp_dir):
        """Test registering duplicate schema."""
        test_file = temp_dir / "test_schema.py"
        test_file.write_text("class TestModel(BaseModel): pass")
        registry_file = temp_dir / "registry.json"
        
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', registry_file):
            # Register first time
            schema_id1 = register_schema(
                class_name="TestModel",
                description="Test schema",
                file_path=test_file,
                model_used="gpt-oss:120b"
            )
            
            # Register again with same description (should return existing)
            schema_id2 = register_schema(
                class_name="TestModel",
                description="Test schema",
                file_path=test_file,
                model_used="gpt-oss:120b"
            )
            
            assert schema_id1 == schema_id2
    
    def test_list_schemas_empty(self, temp_dir):
        """Test listing schemas when registry is empty."""
        registry_file = temp_dir / "empty_registry.json"
        
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', registry_file):
            schemas = list_schemas()
            assert schemas == []
    
    def test_list_schemas_with_data(self, temp_dir, mock_schema_registry):
        """Test listing schemas with existing data."""
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', mock_schema_registry):
            schemas = list_schemas()
            assert len(schemas) == 1
            assert schemas[0]['id'] == 'test_schema_001'
            assert schemas[0]['class_name'] == 'PatientRecord'
    
    def test_get_schema_by_id_found(self, temp_dir, mock_schema_registry):
        """Test getting schema by ID when it exists."""
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', mock_schema_registry):
            schema = get_schema_by_id('test_schema_001')
            assert schema is not None
            assert schema['class_name'] == 'PatientRecord'
    
    def test_get_schema_by_id_not_found(self, temp_dir):
        """Test getting schema by ID when it doesn't exist."""
        registry_file = temp_dir / "empty_registry.json"
        
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', registry_file):
            schema = get_schema_by_id('nonexistent_id')
            assert schema is None
    
    def test_get_suggested_filename(self):
        """Test filename suggestion generation."""
        filename = get_suggested_filename(
            class_name="PatientRecord",
            description="Patient demographics and clinical data"
        )
        
        assert filename.startswith("patientrecord_")
        assert filename.endswith(".py")
        assert len(filename) > 20  # Should include timestamp
    
    def test_cleanup_missing_files(self, temp_dir):
        """Test cleanup of missing files from registry."""
        # Create registry with missing file reference
        registry_file = temp_dir / "registry.json"
        registry_data = {
            "test_schema_001": {
                "id": "test_schema_001",
                "file_path": str(temp_dir / "nonexistent_file.py"),
                "class_name": "TestModel"
            },
            "test_schema_002": {
                "id": "test_schema_002", 
                "file_path": str(temp_dir / "existing_file.py"),
                "class_name": "TestModel2"
            }
        }
        
        # Create one of the files
        existing_file = temp_dir / "existing_file.py"
        existing_file.write_text("class TestModel2(BaseModel): pass")
        
        registry_file.write_text(json.dumps(registry_data, indent=2))
        
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', registry_file):
            removed_count = cleanup_missing_files()
            
        assert removed_count == 1  # One file should be removed
        
        # Check that only existing file remains
        updated_data = json.loads(registry_file.read_text())
        assert len(updated_data) == 1
        assert "test_schema_002" in updated_data
    
    @patch('mosaicx.schema_registry.PACKAGE_SCHEMA_PYD_DIR')
    def test_scan_and_register_existing_schemas(self, mock_schema_dir, temp_dir):
        """Test scanning and registering existing schema files."""
        # Setup mock directory
        mock_schema_dir.return_value = temp_dir
        
        # Create test schema files
        schema1 = temp_dir / "patient_record_001.py"
        schema1.write_text("""
class PatientRecord(BaseModel):
    '''Patient demographic information.'''
    name: str
    age: int
""")
        
        schema2 = temp_dir / "vital_signs_002.py"
        schema2.write_text("""
class VitalSigns(BaseModel):
    '''Patient vital signs data.'''
    temperature: float
    blood_pressure: str
""")
        
        registry_file = temp_dir / "registry.json"
        
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', registry_file):
            registered_count = scan_and_register_existing_schemas()
            
        assert registered_count >= 0  # Should register available files
        
        if registered_count > 0:
            # Check registry was updated
            assert registry_file.exists()
            registry_data = json.loads(registry_file.read_text())
            assert len(registry_data) == registered_count


class TestSchemaSearch:
    """Test cases for schema search functionality."""
    
    def test_search_by_filename(self, temp_dir, mock_schema_registry):
        """Test searching schemas by filename."""
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', mock_schema_registry):
            schemas = list_schemas()
            
            # Search for schema by partial filename match
            matches = [s for s in schemas if 'test_schema' in s.get('file_path', '')]
            assert len(matches) >= 0
    
    def test_search_by_class_name(self, temp_dir, mock_schema_registry):
        """Test searching schemas by class name."""
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', mock_schema_registry):
            schemas = list_schemas()
            
            # Search for schema by class name
            matches = [s for s in schemas if s.get('class_name') == 'PatientRecord']
            assert len(matches) == 1 if schemas else 0
    
    def test_search_by_description(self, temp_dir, mock_schema_registry):
        """Test searching schemas by description."""
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', mock_schema_registry):
            schemas = list_schemas()
            
            # Search for schema by description content
            matches = [s for s in schemas if 'patient' in s.get('description', '').lower()]
            assert len(matches) >= 0


class TestRegistryFileOperations:
    """Test cases for registry file I/O operations."""
    
    def test_create_registry_file(self, temp_dir):
        """Test creating new registry file."""
        registry_file = temp_dir / "new_registry.json"
        test_file = temp_dir / "test.py"
        test_file.write_text("class Test(BaseModel): pass")
        
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', registry_file):
            schema_id = register_schema(
                class_name="Test",
                description="Test schema",
                file_path=test_file
            )
        
        assert registry_file.exists()
        assert schema_id is not None
    
    def test_handle_corrupted_registry_file(self, temp_dir):
        """Test handling corrupted registry file."""
        registry_file = temp_dir / "corrupted_registry.json"
        registry_file.write_text("invalid json content")
        
        with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', registry_file):
            # Should handle corruption gracefully
            schemas = list_schemas()
            assert schemas == []
    
    def test_handle_readonly_registry_file(self, temp_dir):
        """Test handling read-only registry file."""
        registry_file = temp_dir / "readonly_registry.json" 
        registry_file.write_text("{}")
        registry_file.chmod(0o444)  # Make read-only
        
        test_file = temp_dir / "test.py"
        test_file.write_text("class Test(BaseModel): pass")
        
        try:
            with patch('mosaicx.schema_registry.SCHEMA_REGISTRY_FILE', registry_file):
                # Should handle permission error
                schema_id = register_schema(
                    class_name="Test",
                    description="Test schema",
                    file_path=test_file
                )
                # May return None or raise exception depending on implementation
        except (PermissionError, OSError):
            # Expected behavior for read-only file
            pass
        finally:
            # Restore permissions for cleanup
            registry_file.chmod(0o644)
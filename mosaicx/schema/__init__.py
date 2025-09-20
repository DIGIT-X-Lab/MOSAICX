"""
MOSAICX Schema Module

This module contains schema generation, management, and storage functionality.
Includes schema builder and registry components along with generated schemas.
"""

# Import main functions from builder and registry modules
from .builder import synthesize_pydantic_model
from .registry import (
    SchemaRegistry,
    register_schema,
    list_schemas,
    get_schema_by_id,
    get_suggested_filename,
    cleanup_missing_files,
    scan_and_register_existing_schemas
)

# Export public API
__all__ = [
    # Builder functions
    'synthesize_pydantic_model',
    # Registry classes and functions  
    'SchemaRegistry',
    'register_schema',
    'list_schemas',
    'get_schema_by_id',
    'get_suggested_filename',
    'cleanup_missing_files',
    'scan_and_register_existing_schemas'
]

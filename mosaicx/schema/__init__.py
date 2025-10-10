"""
MOSAICX Schema Package - Generation, Registry, and Storage

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Structure first. Insight follows.

Author: Lalith Kumar Shiyam Sundar, PhD
Lab: DIGIT-X Lab
Department: Department of Radiology
University: LMU University Hospital | LMU Munich

Overview:
---------
Bundle together the schema synthesis pipeline, registry tracking, and generated
artifacts that underpin the extraction and summarisation workflows.  Importing
this package exposes helpers that create Pydantic models, manage their
metadata, and locate persisted files within the MOSAICX tree.

Composition:
------------
- ``builder``: Natural-language to Pydantic generation utilities.
- ``registry``: JSON-backed catalogue of schema metadata with housekeeping
  helpers.
- ``templates/python``: Curated Pydantic models bundled with the package.
- ``templates/json``: Placeholder for JSON schema templates shipped alongside the code.
"""

# Import main functions from builder and registry modules
from .builder import synthesize_pydantic_model
from .registry import (
    SchemaRegistry,
    register_schema,
    list_schemas,
    get_schema_by_id,
    get_schema_by_path,
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
    'get_schema_by_path',
    'get_suggested_filename',
    'cleanup_missing_files',
    'scan_and_register_existing_schemas'
]

"""
MOSAICX Document Extraction Module - PDF to Structured Data Pipeline

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligence for X-ray analysis
================================================================================

Overview:
---------
This module provides document extraction capabilities for the MOSAICX application,
leveraging Docling for PDF reading and Instructor + Ollama for structured data 
extraction using generated Pydantic schemas.

Core Functionality:
------------------
• PDF document parsing with Docling
• Schema-based structured extraction using Instructor + Ollama
• Integration with generated Pydantic models from schema_builder
• Error handling and validation for extraction pipeline

Architecture:
------------
The module follows a pipeline pattern: PDF → Text → LLM + Schema → Structured Data
Uses Instructor library for reliable structured outputs from local LLMs.

Usage Examples:
--------------
Extract data from PDF using a schema:
    >>> from mosaicx.extractor import extract_from_pdf
    >>> result = extract_from_pdf("report.pdf", "PatientRecord")
    >>> print(result.name, result.age)

Dependencies:
------------
External Libraries:
    • docling (^2.0.0): Advanced PDF parsing and text extraction
    • instructor (^1.0.0): Structured outputs from LLMs
    • ollama (^0.3.0): Local LLM integration

Module Metadata:
---------------
Author:        Lalith Kumar Shiyam Sundar, PhD
Email:         Lalith.shiyam@med.uni-muenchen.de  
Institution:   DIGIT-X Lab, LMU Radiology | LMU University Hospital
License:       AGPL-3.0 (GNU Affero General Public License v3.0)
Version:       1.0.0
Created:       2025-09-18
Last Modified: 2025-09-18

Copyright Notice:
----------------
© 2025 DIGIT-X Lab, LMU Radiology | LMU University Hospital
This software is distributed under the AGPL-3.0 license.
See LICENSE file for full terms and conditions.
"""

from pathlib import Path
from typing import Any, Dict, Type, Optional, Union
import json
import importlib.util
import sys
import logging

# Suppress noisy logging from Docling and HTTP requests
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling.document_converter").setLevel(logging.WARNING)
logging.getLogger("docling.datamodel.base_models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress HTTP request logs
logging.getLogger("httpcore").setLevel(logging.WARNING)  # Suppress HTTP core logs

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import instructor
from openai import OpenAI
from pydantic import BaseModel

from .constants import (
    DEFAULT_LLM_MODEL,
    PACKAGE_SCHEMA_JSON_DIR,
    PACKAGE_SCHEMA_PYD_DIR,
    MOSAICX_COLORS
)
from .display import styled_message, console


class ExtractionError(Exception):
    """Custom exception for extraction-related errors."""
    pass


def load_schema_model(schema_name: str) -> Type[BaseModel]:
    """
    Load a Pydantic model from the generated schema files.
    
    Args:
        schema_name: Name of the schema model class
        
    Returns:
        The Pydantic model class
        
    Raises:
        ExtractionError: If schema file not found or cannot be loaded
    """
    # Look for Python files in the pyd directory
    pyd_dir = Path(PACKAGE_SCHEMA_PYD_DIR)
    
    # Find files that might contain the schema
    matching_files = []
    for py_file in pyd_dir.glob("*.py"):
        if schema_name.lower() in py_file.name.lower():
            matching_files.append(py_file)
    
    if not matching_files:
        raise ExtractionError(
            f"No schema file found for '{schema_name}' in {pyd_dir}. "
            f"Generate a schema first using: mosaicx generate --desc '...'"
        )
    
    # Use the most recent file
    schema_file = max(matching_files, key=lambda f: f.stat().st_mtime)
    
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("schema_module", schema_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["schema_module"] = module
        spec.loader.exec_module(module)
        
        # Find the schema class
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseModel) and 
                attr.__name__ == schema_name):
                return attr
        
        raise ExtractionError(f"Schema class '{schema_name}' not found in {schema_file}")
        
    except Exception as e:
        raise ExtractionError(f"Failed to load schema from {schema_file}: {e}")


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from PDF using Docling.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        ExtractionError: If PDF cannot be processed
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise ExtractionError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ExtractionError(f"File is not a PDF: {pdf_path}")
    
    try:
        # Initialize Docling converter
        converter = DocumentConverter()
        
        # Convert PDF to document
        result = converter.convert(pdf_path)
        
        # Extract text content
        text_content = result.document.export_to_markdown()
        
        if not text_content.strip():
            raise ExtractionError(f"No text content extracted from {pdf_path}")
        
        return text_content
        
    except Exception as e:
        raise ExtractionError(f"Failed to extract text from PDF {pdf_path}: {e}")


def extract_structured_data(
    text_content: str, 
    schema_class: Type[BaseModel],
    model: str = DEFAULT_LLM_MODEL
) -> BaseModel:
    """
    Extract structured data from text using direct Ollama API calls with fallback.
    
    Args:
        text_content: Input text to extract data from
        schema_class: Pydantic model class for structured output
        model: Ollama model name to use
        
    Returns:
        Instance of the schema class with extracted data
        
    Raises:
        ExtractionError: If extraction fails
    """
    try:
        # First try with Instructor JSON mode
        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,  # Use JSON mode for better compatibility with local models
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data extraction specialist. Extract structured information from documents and return valid JSON only."
                },
                {
                    "role": "user", 
                    "content": f"Extract structured data from this text:\n\n{text_content}"
                }
            ],
            response_model=schema_class,
            max_retries=1,
        )
        return response
        
    except Exception as instructor_error:
        console.print(f"[yellow]Instructor failed: {instructor_error}[/yellow]")
        console.print("[yellow]Falling back to direct Ollama API...[/yellow]")
        
        # Fallback: Direct Ollama API call
        try:
            direct_client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
            
            # Create schema example with proper handling of required vs optional fields
            schema_example = schema_class.model_json_schema()
            example_json = {}
            
            # Get required fields from the schema
            required_fields = schema_example.get("required", [])
            
            for field_name, field_info in schema_example["properties"].items():
                field_type = field_info.get("type")
                enum_values = field_info.get("enum")
                
                if field_name in required_fields:
                    # For required fields, provide meaningful defaults
                    if enum_values:
                        example_json[field_name] = enum_values[0]  # Use first enum value
                    elif field_type == "string":
                        example_json[field_name] = f"example_{field_name}"
                    elif field_type == "number":
                        example_json[field_name] = 1.0
                    elif field_type == "integer":
                        example_json[field_name] = 1
                    elif field_type == "boolean":
                        example_json[field_name] = False
                else:
                    # For optional fields, they can be null or omitted
                    if enum_values:
                        example_json[field_name] = enum_values[0]
                    elif field_type == "string":
                        example_json[field_name] = None
                    elif field_type == "number":
                        example_json[field_name] = None
                    elif field_type == "integer":
                        example_json[field_name] = None
                    elif field_type == "boolean":
                        example_json[field_name] = None
            
            response = direct_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data extraction specialist. You must return ONLY valid JSON that matches the provided schema. No explanation, no markdown, just pure JSON."
                    },
                    {
                        "role": "user", 
                        "content": f"""Extract structured medical data from this text and return as JSON.

Required JSON Schema:
{json.dumps(schema_example, indent=2)}

Example output:
{json.dumps(example_json, indent=2)}

Text to extract from:
{text_content}

Return ONLY the JSON object, nothing else:"""
                    }
                ],
                temperature=0.1,
            )
            
            # Parse the response
            content = response.choices[0].message.content
            if not content:
                raise ExtractionError("Model returned empty content")
            
            # Clean up the response (remove markdown if present)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse and validate the JSON
            try:
                json_data = json.loads(content)
                validated_data = schema_class(**json_data)
                return validated_data
            except json.JSONDecodeError as e:
                raise ExtractionError(f"Invalid JSON from model: {e}\nContent: {content}")
            except Exception as e:
                raise ExtractionError(f"Failed to validate data: {e}")
                
        except Exception as fallback_error:
            raise ExtractionError(f"Both Instructor and direct API failed. Instructor: {instructor_error}, Direct: {fallback_error}")


def extract_from_pdf(
    pdf_path: Union[str, Path],
    schema_name: str,
    model: str = DEFAULT_LLM_MODEL,
    save_result: Optional[Union[str, Path]] = None
) -> BaseModel:
    """
    Complete pipeline: PDF → Text → Structured Data.
    
    Args:
        pdf_path: Path to the PDF file
        schema_name: Name of the schema model class
        model: Ollama model name to use
        save_result: Optional path to save extracted JSON result
        
    Returns:
        Instance of the schema class with extracted data
        
    Raises:
        ExtractionError: If any step in the pipeline fails
    """
    pdf_path = Path(pdf_path)
    
    # Step 1: Load schema model
    with console.status(f"[{MOSAICX_COLORS['info']}]Loading schema model...", spinner="dots"):
        schema_class = load_schema_model(schema_name)
    
    console.print()
    styled_message(f"✨ Schema Model: {schema_class.__name__} ✨", "primary", center=True)
    console.print()
    
    # Step 2: Extract text from PDF
    with console.status(f"[{MOSAICX_COLORS['accent']}]Reading PDF document...", spinner="dots"):
        text_content = extract_text_from_pdf(pdf_path)
    
    # Step 3: Structured extraction
    with console.status(f"[{MOSAICX_COLORS['primary']}]Extracting structured data...", spinner="dots"):
        extracted_data = extract_structured_data(text_content, schema_class, model)
    
    # Step 4: Save result if requested
    if save_result:
        save_path = Path(save_result)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(extracted_data.model_dump(), f, indent=2, default=str)
        
        styled_message(f"💾 Saved result → {save_path.name}", "info", center=True)
    
    return extracted_data

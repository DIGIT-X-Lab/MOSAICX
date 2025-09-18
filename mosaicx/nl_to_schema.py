"""
MOSAICX Natural Language to Schema Module - AI-Powered Schema Generation

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Overview:
---------
This module provides intelligent schema generation capabilities for MOSAICX,
converting natural language descriptions into structured Pydantic models and
JSON schemas. It leverages Large Language Models through Ollama to understand
medical data descriptions and generate appropriate validation schemas.

Core Functionality:
------------------
• Natural language processing for schema induction
• Automatic Pydantic model generation with medical field constraints
• JSON schema generation with validation rules
• Integration with MOSAICX schema registry and file organization
• Robust error handling and schema validation
• Medical domain-specific field type inference

Architecture:
------------
The module uses a structured approach with SchemaSpec as an intermediate
representation that gets compiled into runtime Pydantic models. This ensures
type safety and allows for sophisticated validation rule generation.

Usage Examples:
--------------
Command-line integration:
    $ mosaicx --extract "Patient data with demographics and lab results" --output patient_schema

Programmatic usage:
    >>> from mosaicx.nl_to_schema import generate_schema_from_nl
    >>> schema = generate_schema_from_nl("DICOM metadata fields")
    >>> schema.save_to_registry("medical", "dicom")

Dependencies:
------------
External Libraries:
    • ollama: LLM integration for natural language processing
    • pydantic (^2.0.0): Runtime model compilation and validation
    • rich (^13.0.0): Progress indicators and output formatting

Internal Modules:
    • mosaicx.display: Console output and styling integration
    • mosaicx.schema: Schema registry and file organization

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

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import ollama
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    conint,
    confloat,
    constr,
    create_model,
)

# MOSAICX imports
from display import console, styled_message, MOSAICX_COLORS

# Module metadata
__version__ = "1.0.0"
__author__ = "Lalith Kumar Shiyam Sundar, PhD"
__email__ = "Lalith.shiyam@med.uni-muenchen.de"
__institution__ = "DIGIT-X Lab, LMU Radiology | LMU University Hospital"
__license__ = "AGPL-3.0"
__copyright__ = "© 2025 DIGIT-X Lab, LMU Radiology | LMU University Hospital"

# Export list for public API
__all__ = [
    "SchemaSpec",
    "generate_schema_from_nl",
    "save_schema_to_mosaicx",
    "MosaicXSchemaGenerator"
]

# =============================================================================
# Schema Specification Models
# =============================================================================

Primitive = Literal["string", "integer", "number", "boolean", "date", "datetime"]


class Constraint(BaseModel):
    """Validation constraints for schema fields."""
    regex: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    units: Optional[str] = None  # e.g., "%", "ng/mL"


class EnumSpec(BaseModel):
    """Enumeration specification for controlled vocabularies."""
    name: str
    values: List[str]


class FieldSpec(BaseModel):
    """Field specification for schema generation."""
    name: str
    type: Primitive | Literal["array", "object"]
    description: Optional[str] = None
    required: bool = False
    enum: Optional[str] = None  # reference to EnumSpec.name
    constraints: Optional[Constraint] = None
    items: Optional["FieldSpec"] = None  # for arrays
    properties: Optional[List["FieldSpec"]] = None  # for objects


FieldSpec.model_rebuild()


class SchemaSpec(BaseModel):
    """Complete schema specification generated from natural language."""
    name: str
    version: str = Field(default="1.0.0")
    description: Optional[str] = None
    enums: List[EnumSpec] = Field(default_factory=list)
    fields: List[FieldSpec]


# =============================================================================
# MOSAICX Schema Generator Class
# =============================================================================

class MosaicXSchemaGenerator:
    """
    MOSAICX-integrated schema generator using natural language processing.
    
    This class provides methods to generate Pydantic models and JSON schemas
    from natural language descriptions, with automatic integration into the
    MOSAICX schema registry structure.
    """
    
    def __init__(self, model: str = "gpt-oss:120b", workspace_root: Optional[Path] = None):
        """
        Initialize the schema generator.
        
        Args:
            model (str): Ollama model name for natural language processing
            workspace_root (Path, optional): Root path for MOSAICX workspace
        """
        self.model = model
        self.workspace_root = workspace_root or Path.cwd()
        self.schema_root = self.workspace_root / "mosaicx" / "schema"
        
    def generate_from_description(
        self, 
        description: str, 
        examples: Optional[List[str]] = None,
        max_retries: int = 2
    ) -> SchemaSpec:
        """
        Generate schema specification from natural language description.
        
        Args:
            description (str): Natural language description of the schema
            examples (List[str], optional): Example data to inform schema generation
            max_retries (int): Maximum retry attempts for LLM calls
            
        Returns:
            SchemaSpec: Generated schema specification
        """
        styled_message("Generating schema from natural language...", "info")
        
        try:
            spec = self._induce_schema_with_ollama(description, examples, max_retries)
            styled_message("Schema generation completed successfully", "success")
            return spec
        except Exception as e:
            styled_message(f"Schema generation failed: {e}", "error")
            raise
    
    def save_to_mosaicx_structure(
        self, 
        spec: SchemaSpec, 
        domain: str, 
        schema_name: str
    ) -> Dict[str, Path]:
        """
        Save generated schema to MOSAICX directory structure.
        
        Args:
            spec (SchemaSpec): Generated schema specification
            domain (str): Domain category (e.g., 'medical', 'imaging')
            schema_name (str): Schema file name
            
        Returns:
            Dict[str, Path]: Paths where files were saved
        """
        return save_schema_to_mosaicx(spec, domain, schema_name, self.workspace_root)
    
    def _induce_schema_with_ollama(
        self,
        description: str,
        examples: Optional[List[str]] = None,
        max_retries: int = 2
    ) -> SchemaSpec:
        """Internal method to generate schema using Ollama LLM."""
        system_prompt = self._get_medical_schema_prompt()
        
        examples = examples or []
        user_prompt = f"Natural-language description:\n{description.strip()}\n"
        if examples:
            user_prompt += "\nOptional examples:\n"
            for ex in examples:
                user_prompt += f"\n---\n{ex.strip()}\n---\n"
        
        for attempt in range(max_retries + 1):
            try:
                raw_response = self._call_ollama_robust(system_prompt, user_prompt)
                json_content = self._extract_json_from_response(raw_response)
                
                data = json.loads(json_content)
                return SchemaSpec.model_validate(data)
                
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to generate valid schema after {max_retries + 1} attempts. "
                        f"Last error: {e}"
                    )
                user_prompt += "\nThe previous JSON was invalid. Please provide valid JSON only."
    
    def _get_medical_schema_prompt(self) -> str:
        """Get system prompt optimized for medical schema generation."""
        return """You are a medical data schema expert for MOSAICX. Given a natural-language
description, emit a STRICT JSON object conforming to the SchemaSpec format.
Output ONLY JSON (no markdown, no comments).

Focus on medical data patterns:
- Use appropriate field types for medical measurements, dates, identifiers
- Include units constraints for numerical medical values  
- Consider medical coding systems and controlled vocabularies
- Mark critical medical fields as required
- Use descriptive field names following medical conventions

SchemaSpec JSON format:
{
  "name": str,
  "version": "1.0.0", 
  "description": str | null,
  "enums": [{"name": str, "values": [str, ...]}, ...],
  "fields": [
    {
      "name": str,
      "type": "string"|"integer"|"number"|"boolean"|"date"|"datetime"|"array"|"object",
      "description": str | null,
      "required": bool,
      "enum": str | null,
      "constraints": {"regex": str|null, "minimum": float|null, "maximum": float|null, "units": str|null} | null,
      "items": {FieldSpec} | null,
      "properties": [FieldSpec, ...] | null
    }
  ]
}"""
    
    def _call_ollama_robust(self, system_prompt: str, user_prompt: str) -> str:
        """Robust Ollama API call with fallback strategies."""
        client = ollama.Client()
        
        # Try chat with JSON format first
        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={"temperature": 0}
            )
            return response["message"]["content"]
        except Exception:
            # Fallback to regular chat
            try:
                response = client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={"temperature": 0}
                )
                return response["message"]["content"]
            except Exception as e:
                raise RuntimeError(f"Ollama API call failed: {e}")
    
    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON content from LLM response."""
        text = text.strip()
        
        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Find first complete JSON object
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    return text[start_idx:i + 1]
        
        return text


# =============================================================================
# Utility Functions
# =============================================================================

def _to_camel_case(name: str) -> str:
    """Convert string to CamelCase for class names."""
    parts = re.split(r"[^0-9A-Za-z]+", name)
    return "".join(p.capitalize() for p in parts if p)


def generate_pydantic_code(spec: SchemaSpec) -> str:
    """
    Generate Python code for Pydantic model from schema specification.
    
    Args:
        spec (SchemaSpec): Schema specification
        
    Returns:
        str: Generated Python code
    """
    enum_map = {e.name: e.values for e in spec.enums}
    class_name = _to_camel_case(spec.name)
    
    lines = [
        "from __future__ import annotations",
        "from typing import Any, Dict, List, Optional, Literal", 
        "from pydantic import BaseModel, Field",
        "",
        f'class {class_name}(BaseModel):',
        f'    """{spec.description or spec.name}"""',
        ""
    ]
    
    for field in spec.fields:
        field_type = _get_python_type(field, enum_map)
        default = "..." if field.required else "None"
        description = field.description or ""
        
        if field.constraints and field.constraints.units:
            description += f" [units: {field.constraints.units}]"
        
        lines.append(
            f'    {field.name}: {field_type} = Field({default}, '
            f'description="{description}")'
        )
    
    return "\n".join(lines)


def _get_python_type(field: FieldSpec, enum_map: Dict[str, List[str]]) -> str:
    """Get Python type annotation for a field."""
    if field.enum:
        values = enum_map.get(field.enum, [])
        return "Literal[" + ", ".join(repr(v) for v in values) + "]"
    
    type_map = {
        "string": "str",
        "integer": "int", 
        "number": "float",
        "boolean": "bool",
        "date": "str",
        "datetime": "str"
    }
    
    if field.type in type_map:
        return type_map[field.type]
    elif field.type == "array" and field.items:
        inner_type = _get_python_type(field.items, enum_map)
        return f"List[{inner_type}]"
    elif field.type == "object":
        return "Dict[str, Any]"
    
    return "Any"


def save_schema_to_mosaicx(
    spec: SchemaSpec,
    domain: str, 
    schema_name: str,
    workspace_root: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Save schema specification to MOSAICX directory structure.
    
    Args:
        spec (SchemaSpec): Schema specification to save
        domain (str): Domain category (medical, imaging, analysis)
        schema_name (str): Schema file name
        workspace_root (Path, optional): Workspace root directory
        
    Returns:
        Dict[str, Path]: Dictionary of file type to saved path
    """
    if workspace_root is None:
        workspace_root = Path.cwd()
    
    schema_root = workspace_root / "mosaicx" / "schema"
    domain_path = schema_root / domain
    
    # Create directories if they don't exist
    json_path = domain_path / "json" 
    models_path = domain_path / "models"
    json_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save JSON schema
    json_file = json_path / f"{schema_name}.json"
    with open(json_file, 'w') as f:
        json.dump(spec.model_dump(), f, indent=2)
    saved_files["json"] = json_file
    styled_message(f"JSON schema saved: {json_file}", "success")
    
    # Save Pydantic model
    py_file = models_path / f"{schema_name}.py"
    pydantic_code = generate_pydantic_code(spec)
    with open(py_file, 'w') as f:
        f.write(pydantic_code)
    saved_files["pydantic"] = py_file
    styled_message(f"Pydantic model saved: {py_file}", "success")
    
    return saved_files


def generate_schema_from_nl(
    description: str,
    domain: str,
    schema_name: str, 
    examples: Optional[List[str]] = None,
    model: str = "gpt-oss:120b",
    workspace_root: Optional[Path] = None
) -> Dict[str, Path]:
    """
    High-level function to generate and save schema from natural language.
    
    Args:
        description (str): Natural language description
        domain (str): Schema domain category
        schema_name (str): Name for the generated schema
        examples (List[str], optional): Example data
        model (str): Ollama model to use
        workspace_root (Path, optional): Workspace root directory
        
    Returns:
        Dict[str, Path]: Paths where schema files were saved
    """
    generator = MosaicXSchemaGenerator(model, workspace_root)
    spec = generator.generate_from_description(description, examples)
    return generator.save_to_mosaicx_structure(spec, domain, schema_name)
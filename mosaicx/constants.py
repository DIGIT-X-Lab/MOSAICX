"""
MOSAICX Constants - Centralized Configuration and Metadata

================================================================================
MOSAICX: Medical cOmputational Suite for Advanced Intelligent eXtraction
================================================================================

Overview:
---------
This module defines centralized constants, configuration values, and metadata
used throughout the MOSAICX application. It provides a single source of truth
for version information, author details, branding elements, and default
configuration parameters to ensure consistency across all modules.

Core Contents:
--------------
• Application metadata (version, author, licensing information)
• Institutional attribution and contact details
• Color schemes and branding constants for UI consistency
• Default configuration values for LLM models and processing
• File system paths and naming conventions
• API endpoints and service configuration

Architecture:
------------
Constants are organized into logical groups with clear naming conventions:
- Metadata constants: APPLICATION_*, AUTHOR_*, LICENSE_*
- UI/Display constants: COLORS_*, BANNER_*, STYLE_*  
- Configuration constants: DEFAULT_*, CONFIG_*
- Path constants: PATHS_*, EXTENSIONS_*

Usage Examples:
--------------
Import specific constants:
    >>> from mosaicx.constants import APPLICATION_VERSION, MOSAICX_COLORS
    >>> print(f"MOSAICX v{APPLICATION_VERSION}")

Import all constants:
    >>> from mosaicx import constants
    >>> console.print("Success!", style=constants.MOSAICX_COLORS["success"])

Module Dependencies:
-------------------
Standard Library:
    • pathlib: Path handling for file system constants
    • typing: Type annotations for complex constant structures

Module Metadata:
---------------
Author:        Lalith Kumar Shiyam Sundar, PhD
Email:         Lalith.shiyam@med.uni-muenchen.de  
Institution:   DIGIT-X Lab, LMU Radiology | LMU University Hospital
License:       AGPL-3.0 (GNU Affero General Public License v3.0)
Version:       (dynamic via APPLICATION_VERSION)
Created:       2025-09-18
Last Modified: 2025-09-18

Copyright Notice:
----------------
© 2025 DIGIT-X Lab, LMU Radiology | LMU University Hospital
This software is distributed under the AGPL-3.0 license.
See LICENSE file for full terms and conditions.
"""

from pathlib import Path
from typing import Dict, List

from importlib.metadata import PackageNotFoundError, version as pkg_version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]


def _load_application_version() -> str:
    """Resolve the application version from package metadata or pyproject."""

    try:
        return pkg_version("mosaicx")
    except PackageNotFoundError:
        project_root = Path(__file__).resolve().parent.parent
        pyproject_path = project_root / "pyproject.toml"
        if tomllib is not None and pyproject_path.exists():
            try:
                with pyproject_path.open("rb") as handle:
                    data = tomllib.load(handle)
                return data.get("project", {}).get("version", "0.0.0")
            except (tomllib.TOMLDecodeError, AttributeError):
                pass

    return "0.0.0"

# =============================================================================
# APPLICATION METADATA
# =============================================================================

APPLICATION_NAME = "MOSAICX"
APPLICATION_FULL_NAME = "Medical cOmputational Suite for Advanced Intelligent eXtraction"
APPLICATION_VERSION = _load_application_version()
APPLICATION_DESCRIPTION = (
    "Intelligent radiology report extraction using local LLMs for medical data structuring"
)

# =============================================================================
# AUTHOR & INSTITUTIONAL INFORMATION
# =============================================================================

AUTHOR_NAME = "Lalith Kumar Shiyam Sundar, PhD"
AUTHOR_EMAIL = "Lalith.shiyam@med.uni-muenchen.de"
INSTITUTION_NAME = "DIGIT-X Lab, LMU Radiology | LMU University Hospital"
INSTITUTION_SHORT = "DIGIT-X Lab, LMU"

# =============================================================================
# LICENSING & COPYRIGHT
# =============================================================================

LICENSE_TYPE = "AGPL-3.0"
LICENSE_FULL_NAME = "GNU Affero General Public License v3.0"
COPYRIGHT_YEAR = "2025"
COPYRIGHT_HOLDER = "DIGIT-X Lab, LMU Radiology | LMU University Hospital"
COPYRIGHT_NOTICE = f"© {COPYRIGHT_YEAR} {COPYRIGHT_HOLDER}"

# =============================================================================
# UI COLORS & BRANDING
# =============================================================================

MOSAICX_COLORS: Dict[str, str] = {
    "primary": "#ff79c6",      # Dracula Pink - vibrant primary
    "secondary": "#6272a4",    # Dracula Comment - elegant secondary
    "success": "#50fa7b",      # Dracula Green - success
    "warning": "#f1fa8c",      # Dracula Yellow - warning
    "error": "#ff5555",        # Dracula Red - error
    "info": "#8be9fd",         # Dracula Cyan - info
    "accent": "#bd93f9",       # Dracula Purple - accent
    "muted": "#44475a"         # Dracula Current Line - muted text
}

# Banner and display configuration
BANNER_STYLE = "block"
BANNER_COLORS = [MOSAICX_COLORS["secondary"], MOSAICX_COLORS["primary"]]  # Overlay1 to Peach gradient

# =============================================================================
# SCHEMA GENERATION PROMPTS
# =============================================================================

# System prompt for LLM-based schema generation
SCHEMA_GENERATION_SYSTEM_PROMPT = """You are a precise schema designer. Given a natural-language
description (and optional example reports), emit a STRICT JSON object that conforms
EXACTLY to the 'SchemaSpec' shape below. Output ONLY JSON (no markdown, no comments).

Rules:
- Allowed field types: "string" | "integer" | "number" | "boolean" | "date" | "datetime" | "array" | "object".
- For arrays, include "items"; for objects include "properties".
- Mark required fields with "required": true; others false or omit.
- Use "enums" to define allowed values; reference via field.enum.
- Include short 'description' for each field when clear.
- Use 'constraints' (pattern/minimum/maximum/units) when the description implies them.
- If uncertain, include the field but set required=false.

SchemaSpec (JSON shape):
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
      "constraints": { "pattern": str|null, "minimum": float|null, "maximum": float|null, "units": str|null } | null,
      "items": { FieldSpec } | null,
      "properties": [ FieldSpec, ... ] | null
    }
  ]
}
"""

# =============================================================================
# DEFAULT CONFIGURATION VALUES
# =============================================================================

# LLM Model defaults
DEFAULT_LLM_MODEL = "gpt-oss:120b"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_RETRIES = 2

# Schema generation defaults
DEFAULT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_FIELD_TYPES = [
    "string", "integer", "number", "boolean", 
    "date", "datetime", "array", "object"
]

# File processing defaults
DEFAULT_OUTPUT_FORMAT = "json"
SUPPORTED_FORMATS = ["json", "yaml", "py"]
DEFAULT_ENCODING = "utf-8"

# =============================================================================
# PATH CONSTANTS
# =============================================================================

# File extensions
SCHEMA_EXTENSIONS = {
    "json": ".json",
    "yaml": ".yaml", 
    "python": ".py"
}

# Default directories (relative to project root)
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_SCHEMA_DIR = "schemas"
DEFAULT_MODELS_DIR = "models"

# Schema-specific directories within the package
PACKAGE_SCHEMA_DIR = "mosaicx/schema"
PACKAGE_SCHEMA_JSON_DIR = "mosaicx/schema/json"
PACKAGE_SCHEMA_PYD_DIR = "mosaicx/schema/pyd"

# =============================================================================
# API & SERVICE CONFIGURATION
# =============================================================================

# Ollama service defaults
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 300  # 5 minutes

# Request configuration
DEFAULT_REQUEST_TIMEOUT = 60
MAX_REQUEST_RETRIES = 3

# =============================================================================
# EXPORT LIST
# =============================================================================

__all__ = [
    # Application metadata
    "APPLICATION_NAME",
    "APPLICATION_FULL_NAME", 
    "APPLICATION_VERSION",
    "APPLICATION_DESCRIPTION",
    
    # Author information
    "AUTHOR_NAME",
    "AUTHOR_EMAIL", 
    "INSTITUTION_NAME",
    "INSTITUTION_SHORT",
    
    # Licensing
    "LICENSE_TYPE",
    "LICENSE_FULL_NAME",
    "COPYRIGHT_YEAR",
    "COPYRIGHT_HOLDER", 
    "COPYRIGHT_NOTICE",
    
    # UI & Branding
    "MOSAICX_COLORS",
    "BANNER_STYLE",
    "BANNER_COLORS",
    
    # Configuration
    "DEFAULT_LLM_MODEL",
    "DEFAULT_TEMPERATURE", 
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_SCHEMA_VERSION",
    "SUPPORTED_FIELD_TYPES",
    "DEFAULT_OUTPUT_FORMAT",
    "SUPPORTED_FORMATS",
    "DEFAULT_ENCODING",
    "SCHEMA_GENERATION_SYSTEM_PROMPT",
    
    # Paths
    "SCHEMA_EXTENSIONS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SCHEMA_DIR", 
    "DEFAULT_MODELS_DIR",
    "PACKAGE_SCHEMA_DIR",
    "PACKAGE_SCHEMA_JSON_DIR",
    "PACKAGE_SCHEMA_PYD_DIR",
    
    # API Configuration
    "DEFAULT_OLLAMA_HOST",
    "OLLAMA_TIMEOUT",
    "DEFAULT_REQUEST_TIMEOUT",
    "MAX_REQUEST_RETRIES"
]

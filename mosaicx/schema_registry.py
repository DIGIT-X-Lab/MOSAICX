"""
MOSAICX Schema Registry - Schema Management and Tracking

This module provides functionality to register, track, and manage generated
Pydantic schemas, making it easy for users to identify and select the right
schema for their needs.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

from .constants import PACKAGE_SCHEMA_PYD_DIR


class SchemaRegistry:
    """Manages a registry of generated Pydantic schemas."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize the schema registry.
        
        Args:
            registry_path: Path to the registry JSON file. If None, uses default location.
        """
        if registry_path is None:
            registry_path = Path(PACKAGE_SCHEMA_PYD_DIR) / "schema_registry.json"
        
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or create new one
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the registry from file or create an empty one."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If registry is corrupted, start fresh
                return {"schemas": {}, "version": "1.0.0"}
        else:
            return {"schemas": {}, "version": "1.0.0"}
    
    def _save_registry(self) -> None:
        """Save the registry to file."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self._registry, f, indent=2, default=str)
        except IOError as e:
            print(f"Warning: Could not save schema registry: {e}")
    
    def _generate_description_hash(self, description: str) -> str:
        """Generate a short hash from the description for grouping."""
        return hashlib.md5(description.lower().strip().encode()).hexdigest()[:8]
    
    def register_schema(
        self,
        class_name: str,
        description: str,
        file_path: Path,
        model_used: str,
        temperature: float = 0.2
    ) -> str:
        """Register a new generated schema.
        
        Args:
            class_name: Name of the Pydantic class
            description: Natural language description used to generate the schema
            file_path: Path to the generated Python file
            model_used: LLM model used for generation
            temperature: Temperature setting used
            
        Returns:
            Schema ID for referencing this schema
        """
        timestamp = datetime.now().isoformat()
        description_hash = self._generate_description_hash(description)
        schema_id = f"{class_name.lower()}_{description_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        schema_entry = {
            "id": schema_id,
            "class_name": class_name,
            "description": description,
            "description_hash": description_hash,
            "file_path": str(file_path),
            "file_name": file_path.name,
            "model_used": model_used,
            "temperature": temperature,
            "created_at": timestamp,
            "file_exists": file_path.exists()
        }
        
        # Store in registry
        if "schemas" not in self._registry:
            self._registry["schemas"] = {}
        
        self._registry["schemas"][schema_id] = schema_entry
        self._save_registry()
        
        return schema_id
    
    def list_schemas(
        self, 
        class_name_filter: Optional[str] = None,
        description_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all registered schemas with optional filtering.
        
        Args:
            class_name_filter: Filter by class name (case-insensitive partial match)
            description_filter: Filter by description (case-insensitive partial match)
            
        Returns:
            List of schema entries
        """
        schemas = list(self._registry.get("schemas", {}).values())
        
        # Apply filters
        if class_name_filter:
            schemas = [s for s in schemas if class_name_filter.lower() in s["class_name"].lower()]
        
        if description_filter:
            schemas = [s for s in schemas if description_filter.lower() in s["description"].lower()]
        
        # Sort by creation date (newest first)
        schemas.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Update file_exists status
        for schema in schemas:
            schema["file_exists"] = Path(schema["file_path"]).exists()
        
        return schemas
    
    def get_schema_by_id(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific schema by its ID."""
        schema = self._registry.get("schemas", {}).get(schema_id)
        if schema:
            schema["file_exists"] = Path(schema["file_path"]).exists()
        return schema
    
    def get_schemas_by_class_name(self, class_name: str) -> List[Dict[str, Any]]:
        """Get all schemas with a specific class name."""
        return [
            schema for schema in self._registry.get("schemas", {}).values()
            if schema["class_name"].lower() == class_name.lower()
        ]
    
    def cleanup_missing_files(self) -> int:
        """Remove registry entries for files that no longer exist.
        
        Returns:
            Number of entries removed
        """
        schemas = self._registry.get("schemas", {})
        removed_count = 0
        
        schema_ids_to_remove = []
        for schema_id, schema in schemas.items():
            if not Path(schema["file_path"]).exists():
                schema_ids_to_remove.append(schema_id)
        
        for schema_id in schema_ids_to_remove:
            del schemas[schema_id]
            removed_count += 1
        
        if removed_count > 0:
            self._save_registry()
        
        return removed_count
    
    def get_suggested_filename(self, class_name: str, description: str) -> str:
        """Generate a suggested filename that includes context from description.
        
        Args:
            class_name: The Pydantic class name
            description: Natural language description
            
        Returns:
            Suggested filename with timestamp
        """
        # Extract key words from description for filename
        import re
        
        # Get important words (remove common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', description.lower())
        key_words = [w for w in words if len(w) > 2 and w not in stop_words][:3]  # Take first 3 meaningful words
        
        # Create descriptive part
        if key_words:
            desc_part = '_'.join(key_words)
        else:
            desc_part = 'schema'
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{class_name.lower()}_{desc_part}_{timestamp}.py"


# Global registry instance
_registry = SchemaRegistry()


def register_schema(class_name: str, description: str, file_path: Path, model_used: str, temperature: float = 0.2) -> str:
    """Register a new schema in the global registry."""
    return _registry.register_schema(class_name, description, file_path, model_used, temperature)


def list_schemas(class_name_filter: Optional[str] = None, description_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """List schemas from the global registry."""
    return _registry.list_schemas(class_name_filter, description_filter)


def get_schema_by_id(schema_id: str) -> Optional[Dict[str, Any]]:
    """Get schema by ID from the global registry."""
    return _registry.get_schema_by_id(schema_id)


def get_suggested_filename(class_name: str, description: str) -> str:
    """Get suggested filename from the global registry."""
    return _registry.get_suggested_filename(class_name, description)


def cleanup_missing_files() -> int:
    """Cleanup missing files from the global registry."""
    return _registry.cleanup_missing_files()


def scan_and_register_existing_schemas() -> int:
    """Scan the schema directory and register any untracked schema files.
    
    This is useful for migrating existing schemas that were created before
    the registry system was implemented.
    
    Returns:
        Number of new schemas registered
    """
    import re
    from pathlib import Path
    
    schema_dir = Path(PACKAGE_SCHEMA_PYD_DIR)
    if not schema_dir.exists():
        return 0
    
    registered_count = 0
    existing_files = {schema['file_name'] for schema in _registry.list_schemas()}
    
    # Scan all .py files in the schema directory
    for py_file in schema_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        # Skip if already registered
        if py_file.name in existing_files:
            continue
        
        try:
            # Try to extract class name and description from the file
            class_name, description = _extract_schema_info_from_file(py_file)
            
            if class_name and description:
                # Register the schema with estimated metadata
                schema_id = _registry.register_schema(
                    class_name=class_name,
                    description=description,
                    file_path=py_file,
                    model_used="unknown",  # Can't determine from existing files
                    temperature=0.2  # Default value
                )
                registered_count += 1
                print(f"Registered existing schema: {py_file.name} -> {schema_id}")
            else:
                print(f"Could not extract info from: {py_file.name}")
                
        except Exception as e:
            print(f"Error processing {py_file.name}: {e}")
            continue
    
    return registered_count


def _extract_schema_info_from_file(file_path: Path) -> tuple[str | None, str | None]:
    """Extract class name and description from a schema file.
    
    Args:
        file_path: Path to the schema file
        
    Returns:
        Tuple of (class_name, description) or (None, None) if extraction fails
    """
    import re
    
    try:
        content = file_path.read_text()
        
        # Extract class name using regex
        class_match = re.search(r'class\s+(\w+)\s*\(BaseModel\)', content)
        if not class_match:
            return None, None
        
        class_name = class_match.group(1)
        
        # Extract description from class docstring
        docstring_match = re.search(r'class\s+\w+\s*\(BaseModel\):\s*"""([^"]+)"""', content, re.DOTALL)
        if docstring_match:
            description = docstring_match.group(1).strip()
            # Clean up the description - take first line if multiline
            description = description.split('\n')[0].strip()
        else:
            # Fallback: try to infer from filename
            base_name = file_path.stem
            # Remove timestamp part
            clean_name = re.sub(r'_\d{8}_\d{6}$', '', base_name)
            description = f"Generated schema for {clean_name}"
        
        return class_name, description
        
    except Exception:
        return None, None
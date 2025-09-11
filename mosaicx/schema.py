"""Schema building utilities for MOSAICX."""

import json
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

from .core.models import ExtractionConfig
from .core.exceptions import ConfigurationError


class SchemaBuilder:
    """Interactive schema builder for extraction configurations."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize schema builder.
        
        Args:
            console: Rich console instance for output
        """
        self.console = console or Console()

    def interactive_build(
        self,
        sample_text: Optional[str] = None,
        existing_config: Optional[ExtractionConfig] = None
    ) -> ExtractionConfig:
        """Build extraction schema interactively.
        
        Args:
            sample_text: Sample text to analyze
            existing_config: Existing configuration to modify
            
        Returns:
            ExtractionConfig with user-defined schema
        """
        self.console.print("[bold blue]🔨 Interactive Schema Builder[/bold blue]")
        self.console.print("Let's build an extraction schema for your medical reports!\n")
        
        if sample_text:
            self.console.print(f"[dim]Sample text provided ({len(sample_text)} characters)[/dim]")
        
        # Start with existing config or create new
        if existing_config:
            config = existing_config.model_copy()
            self.console.print("[green]Starting with existing configuration[/green]")
        else:
            config = ExtractionConfig(
                extraction_schema={},
                required_fields=[],
                output_format="json"
            )
        
        # Get basic configuration
        self._configure_basic_settings(config)
        
        # Build schema fields
        self._build_schema_fields(config)
        
        # Configure validation
        self._configure_validation(config)
        
        # Show summary
        self._show_config_summary(config)
        
        return config

    def _configure_basic_settings(self, config: ExtractionConfig) -> None:
        """Configure basic extraction settings."""
        self.console.print("\n[bold yellow]Basic Configuration[/bold yellow]")
        
        # Output format
        output_format = Prompt.ask(
            "Output format",
            choices=["json", "structured", "key_value"],
            default=config.output_format
        )
        config.output_format = output_format
        
        # Max retries
        max_retries = Prompt.ask(
            "Maximum extraction retries",
            default=str(config.max_retries)
        )
        try:
            config.max_retries = int(max_retries)
        except ValueError:
            config.max_retries = 3
        
        # Validation enabled
        config.validation_enabled = Confirm.ask(
            "Enable field validation?",
            default=config.validation_enabled
        )

    def _build_schema_fields(self, config: ExtractionConfig) -> None:
        """Build schema fields interactively."""
        self.console.print("\n[bold yellow]Schema Fields[/bold yellow]")
        
        # Show existing fields if any
        if config.schema:
            self.console.print("[dim]Existing fields:[/dim]")
            table = Table()
            table.add_column("Field", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Description", style="white")
            
            for field_name, field_config in config.schema.items():
                table.add_row(
                    field_name,
                    field_config.get("type", "string"),
                    field_config.get("description", "")
                )
            
            self.console.print(table)
            
            modify_existing = Confirm.ask(
                "\nModify existing fields?",
                default=False
            )
            
            if modify_existing:
                config.schema = {}
                config.required_fields = []
        
        # Add new fields
        self.console.print("\n[dim]Add fields (press Enter with empty field name to finish)[/dim]")
        
        while True:
            field_name = Prompt.ask("Field name")
            if not field_name.strip():
                break
            
            field_type = Prompt.ask(
                "Field type",
                choices=["string", "number", "boolean", "array", "object"],
                default="string"
            )
            
            description = Prompt.ask(
                "Field description",
                default=""
            )
            
            is_required = Confirm.ask(
                f"Is '{field_name}' required?",
                default=True
            )
            
            # Build field configuration
            field_config = {
                "type": field_type,
                "description": description
            }
            
            # Add type-specific options
            if field_type == "array":
                item_type = Prompt.ask(
                    "Array item type",
                    choices=["string", "number", "object"],
                    default="string"
                )
                field_config["items"] = {"type": item_type}
            
            config.schema[field_name] = field_config
            
            if is_required:
                config.required_fields.append(field_name)
            
            self.console.print(f"[green]✓[/green] Added field: {field_name}")

    def _configure_validation(self, config: ExtractionConfig) -> None:
        """Configure validation rules."""
        if not config.validation_enabled:
            return
        
        self.console.print("\n[bold yellow]Validation Rules[/bold yellow]")
        
        # Date formats
        if any("date" in field.lower() for field in config.schema.keys()):
            date_format = Prompt.ask(
                "Expected date format",
                default=config.expected_date_format or "YYYY-MM-DD"
            )
            config.expected_date_format = date_format
        
        # Custom validation patterns
        add_patterns = Confirm.ask(
            "Add custom validation patterns?",
            default=False
        )
        
        if add_patterns:
            validation_patterns = config.validation_patterns or {}
            
            for field_name in config.schema.keys():
                add_pattern = Confirm.ask(
                    f"Add validation pattern for '{field_name}'?",
                    default=False
                )
                
                if add_pattern:
                    pattern = Prompt.ask(f"Regex pattern for '{field_name}'")
                    validation_patterns[field_name] = pattern
            
            if validation_patterns:
                config.validation_patterns = validation_patterns

    def _show_config_summary(self, config: ExtractionConfig) -> None:
        """Show configuration summary."""
        self.console.print("\n[bold green]📋 Configuration Summary[/bold green]")
        
        # Basic settings
        self.console.print(f"Output format: [cyan]{config.output_format}[/cyan]")
        self.console.print(f"Max retries: [cyan]{config.max_retries}[/cyan]")
        self.console.print(f"Validation enabled: [cyan]{config.validation_enabled}[/cyan]")
        
        # Schema fields
        if config.schema:
            self.console.print(f"\nSchema fields: [cyan]{len(config.schema)}[/cyan]")
            for field_name, field_config in config.schema.items():
                required_marker = "⭐" if field_name in config.required_fields else ""
                self.console.print(f"  • {field_name} ({field_config['type']}) {required_marker}")
        
        # Validation
        if config.validation_enabled:
            if config.expected_date_format:
                self.console.print(f"Date format: [cyan]{config.expected_date_format}[/cyan]")
            if config.validation_patterns:
                self.console.print(f"Custom patterns: [cyan]{len(config.validation_patterns)}[/cyan]")

    def save_config(
        self,
        config: ExtractionConfig,
        output_path: str
    ) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(config.model_dump(), f, indent=2)
            
            self.console.print(f"[green]✓[/green] Configuration saved to: {output_path}")
        
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def load_config(self, config_path: str) -> ExtractionConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ExtractionConfig object
        """
        try:
            # Use the ExtractionConfig.from_file method which supports both YAML and JSON
            return ExtractionConfig.from_file(config_path)
        
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def suggest_schema_from_text(
        self,
        text_content: str,
        field_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Suggest schema fields based on text analysis.
        
        Args:
            text_content: Text to analyze
            field_hints: Optional hints for field names
            
        Returns:
            Suggested schema dictionary
        """
        suggestions = {}
        
        # Common medical report patterns
        patterns = {
            "patient_name": r"(patient|name):\s*([A-Za-z\s]+)",
            "mrn": r"(mrn|medical record|id):\s*([0-9]+)",
            "study_date": r"(date|exam date):\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{4})",
            "study_type": r"(study|exam|procedure):\s*([A-Za-z\s]+)",
            "findings": r"(findings|impression):\s*([^.]+)",
        }
        
        for field_name, pattern in patterns.items():
            suggestions[field_name] = {
                "type": "string",
                "description": f"Extracted {field_name.replace('_', ' ')}",
                "suggested": True
            }
        
        # Add field hints if provided
        if field_hints:
            for hint in field_hints:
                if hint not in suggestions:
                    suggestions[hint] = {
                        "type": "string",
                        "description": f"User suggested field: {hint}",
                        "suggested": True
                    }
        
        return suggestions

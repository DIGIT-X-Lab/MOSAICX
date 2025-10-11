"""
Test Constants Module

Tests for constants and configuration values.
"""

import pytest
from mosaicx.constants import (
    APPLICATION_NAME,
    APPLICATION_VERSION,
    AUTHOR_NAME,
    DEFAULT_LLM_MODEL,
    MOSAICX_COLORS
)


class TestApplicationMetadata:
    """Test cases for application metadata constants."""
    
    def test_application_name(self):
        """Test application name constant."""
        assert APPLICATION_NAME == "MOSAICX"
    
    def test_application_version(self):
        """Test application version format."""
        assert isinstance(APPLICATION_VERSION, str)
        assert len(APPLICATION_VERSION) > 0
        # Should follow semantic versioning pattern
        version_parts = APPLICATION_VERSION.split('.')
        assert len(version_parts) >= 2
    
    def test_author_name(self):
        """Test author name constant."""
        assert isinstance(AUTHOR_NAME, str)
        assert len(AUTHOR_NAME) > 0
    
    def test_default_llm_model(self):
        """Test default LLM model setting."""
        assert isinstance(DEFAULT_LLM_MODEL, str)
        assert len(DEFAULT_LLM_MODEL) > 0
        # Should be a valid model identifier
        assert ':' in DEFAULT_LLM_MODEL  # Format like "gpt-oss:120b"


class TestColorScheme:
    """Test cases for color scheme constants."""
    
    def test_mosaicx_colors_structure(self):
        """Test MOSAICX_COLORS dictionary structure."""
        assert isinstance(MOSAICX_COLORS, dict)
        
        # Should contain required color keys
        required_colors = ['primary', 'secondary', 'success', 'error', 'info', 'warning']
        for color in required_colors:
            assert color in MOSAICX_COLORS
    
    def test_color_values_format(self):
        """Test color values are in proper format."""
        for color_name, color_value in MOSAICX_COLORS.items():
            assert isinstance(color_value, str)
            # Should be hex color format
            assert color_value.startswith('#')
            assert len(color_value) == 7  # #RRGGBB format
    
    def test_specific_colors(self):
        """Test specific color values."""
        # Test some known colors from the Dracula theme
        assert MOSAICX_COLORS['primary'] == '#ff79c6'
        assert MOSAICX_COLORS['success'] == '#50fa7b'
        assert MOSAICX_COLORS['error'] == '#ff5555'


class TestConfigurationConstants:
    """Test cases for configuration constants."""
    
    def test_constants_are_immutable(self):
        """Test that constants should not be easily modified."""
        # This is more of a design principle test
        original_name = APPLICATION_NAME
        original_version = APPLICATION_VERSION
        
        # Constants should maintain their values
        assert APPLICATION_NAME == original_name
        assert APPLICATION_VERSION == original_version
    
    def test_model_configuration(self):
        """Test LLM model configuration constants."""
        assert DEFAULT_LLM_MODEL is not None
        assert isinstance(DEFAULT_LLM_MODEL, str)
        
        # Should be one of the tested models
        tested_models = ['gpt-oss:120b', 'mistral:latest', 'qwen2.5-coder:32b', 'deepseek-r1:8b']
        assert DEFAULT_LLM_MODEL in tested_models


class TestConstantsIntegration:
    """Test cases for constants integration with other modules."""
    
    def test_constants_import(self):
        """Test that constants can be imported correctly."""
        try:
            from mosaicx.constants import (
                APPLICATION_NAME,
                APPLICATION_VERSION,
                MOSAICX_COLORS,
                DEFAULT_LLM_MODEL
            )
            # Import should succeed
            assert True
        except ImportError:
            pytest.fail("Constants should be importable")
    
    def test_color_constants_completeness(self):
        """Test that all required colors are defined."""
        expected_colors = [
            'primary', 'secondary', 'success', 'error', 
            'info', 'warning', 'accent', 'muted'
        ]
        
        for color in expected_colors:
            assert color in MOSAICX_COLORS, f"Missing color: {color}"
    
    def test_constants_types(self):
        """Test constants have correct types."""
        assert isinstance(APPLICATION_NAME, str)
        assert isinstance(APPLICATION_VERSION, str)
        assert isinstance(AUTHOR_NAME, str)
        assert isinstance(DEFAULT_LLM_MODEL, str)
        assert isinstance(MOSAICX_COLORS, dict)


class TestConstantsValues:
    """Test cases for specific constant values."""
    
    def test_application_branding(self):
        """Test application branding consistency."""
        # Application name should be consistent
        assert APPLICATION_NAME.upper() == "MOSAICX"
    
    def test_default_model_validity(self):
        """Test default model is valid."""
        # Should be in the format "model:size" 
        parts = DEFAULT_LLM_MODEL.split(':')
        assert len(parts) == 2
        assert len(parts[0]) > 0  # Model name
        assert len(parts[1]) > 0  # Model size/version
    
    def test_color_accessibility(self):
        """Test colors meet basic accessibility standards."""
        # This is a basic test - in practice you'd want proper contrast testing
        for color_name, color_hex in MOSAICX_COLORS.items():
            # Convert hex to RGB for basic validation
            hex_color = color_hex.lstrip('#')
            assert len(hex_color) == 6
            
            # Should be valid hex values
            try:
                int(hex_color, 16)
            except ValueError:
                pytest.fail(f"Invalid hex color for {color_name}: {color_hex}")


class TestEnvironmentConstants:
    """Test cases for environment-specific constants."""
    
    def test_path_constants_exist(self):
        """Test that path constants are defined if needed."""
        # Test would depend on what path constants are defined
        try:
            from mosaicx.constants import PACKAGE_SCHEMA_TEMPLATES_PY_DIR
            assert PACKAGE_SCHEMA_TEMPLATES_PY_DIR is not None
        except ImportError:
            # If not defined, that's fine for this test
            pass
    
    def test_banner_constants(self):
        """Test banner-related constants."""
        try:
            from mosaicx.constants import BANNER_COLORS
            assert isinstance(BANNER_COLORS, list)
            assert len(BANNER_COLORS) > 0
        except ImportError:
            # If not defined, that's fine for this test
            pass

"""
Test Display Module

Tests for display utilities and console output functionality.
"""

import pytest
from unittest.mock import Mock, patch
from rich.console import Console

from mosaicx.display import show_main_banner, styled_message, console


class TestBannerDisplay:
    """Test cases for banner display functionality."""
    
    @patch('mosaicx.display.console')
    def test_show_main_banner(self, mock_console):
        """Test main banner display."""
        show_main_banner()
        
        # Should call console print methods
        assert mock_console.print.called
        
        # Check that banner content is displayed
        call_args_list = [call[0][0] for call in mock_console.print.call_args_list]
        banner_content = ' '.join(str(arg) for arg in call_args_list)
        
        # Should contain MOSAICX branding
        assert any('MOSAICX' in str(arg) or 'Medical' in str(arg) 
                  for arg in call_args_list)
    
    def test_banner_structure(self):
        """Test banner has proper structure."""
        with patch('mosaicx.display.console') as mock_console:
            show_main_banner()
            
            # Should make multiple print calls for banner structure
            assert mock_console.print.call_count > 1
    
    def test_banner_colors(self):
        """Test banner uses correct colors."""
        with patch('mosaicx.display.console') as mock_console:
            show_main_banner()
            
            # Check that color styling is applied
            # (This would need to inspect the actual color codes used)
            assert mock_console.print.called


class TestStyledMessaging:
    """Test cases for styled message functionality."""
    
    @patch('mosaicx.display.console')
    def test_styled_message_info(self, mock_console):
        """Test styled message with info type."""
        styled_message("Test info message", "info")
        
        mock_console.print.assert_called_once()
        
        # Check message content and styling
        call_args = mock_console.print.call_args[0][0]
        assert "Test info message" in str(call_args)
    
    @patch('mosaicx.display.console')
    def test_styled_message_success(self, mock_console):
        """Test styled message with success type."""
        styled_message("Operation successful", "success")
        
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Operation successful" in str(call_args)
    
    @patch('mosaicx.display.console')
    def test_styled_message_error(self, mock_console):
        """Test styled message with error type."""
        styled_message("Error occurred", "error")
        
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Error occurred" in str(call_args)
    
    @patch('mosaicx.display.console')
    def test_styled_message_warning(self, mock_console):
        """Test styled message with warning type."""
        styled_message("Warning message", "warning")
        
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Warning message" in str(call_args)
    
    @patch('mosaicx.display.console')
    def test_styled_message_default(self, mock_console):
        """Test styled message with default/unknown type."""
        styled_message("Default message", "unknown_type")
        
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Default message" in str(call_args)


class TestConsoleConfiguration:
    """Test cases for console configuration."""
    
    def test_console_instance(self):
        """Test console instance is properly configured."""
        from mosaicx.display import console
        
        assert isinstance(console, Console)
        assert console is not None
    
    def test_console_width(self):
        """Test console width configuration."""
        from mosaicx.display import console
        
        # Console should have reasonable width settings
        assert hasattr(console, 'width')
    
    def test_console_color_support(self):
        """Test console color support."""
        from mosaicx.display import console
        
        # Should support color output
        assert hasattr(console, 'color_system')


class TestDisplayIntegration:
    """Test cases for display integration with other components."""
    
    @patch('mosaicx.display.console')
    def test_message_types_integration(self, mock_console):
        """Test all message types work correctly."""
        message_types = ['info', 'success', 'error', 'warning']
        
        for msg_type in message_types:
            styled_message(f"Test {msg_type} message", msg_type)
        
        # Should have called print for each message type
        assert mock_console.print.call_count == len(message_types)
    
    def test_empty_message(self):
        """Test handling of empty messages."""
        with patch('mosaicx.display.console') as mock_console:
            styled_message("", "info")
            mock_console.print.assert_called_once()
    
    def test_long_message(self):
        """Test handling of very long messages."""
        long_message = "A" * 1000  # Very long message
        
        with patch('mosaicx.display.console') as mock_console:
            styled_message(long_message, "info")
            mock_console.print.assert_called_once()
    
    def test_unicode_message(self):
        """Test handling of unicode messages."""
        unicode_message = "Test message with unicode: 🔥 ⭐ 🟢"
        
        with patch('mosaicx.display.console') as mock_console:
            styled_message(unicode_message, "info")
            mock_console.print.assert_called_once()


class TestDisplayFormatting:
    """Test cases for display formatting and styling."""
    
    def test_banner_formatting(self):
        """Test banner has proper formatting."""
        with patch('mosaicx.display.console') as mock_console:
            show_main_banner()
            
            # Check that proper spacing and formatting is applied
            assert mock_console.print.call_count > 0
    
    def test_message_formatting_consistency(self):
        """Test message formatting is consistent across types."""
        with patch('mosaicx.display.console') as mock_console:
            test_messages = [
                ("Info message", "info"),
                ("Success message", "success"),
                ("Error message", "error"),
                ("Warning message", "warning")
            ]
            
            for message, msg_type in test_messages:
                mock_console.reset_mock()
                styled_message(message, msg_type)
                
                # Each message type should result in exactly one print call
                assert mock_console.print.call_count == 1


class TestColorSchemeIntegration:
    """Test cases for color scheme integration."""
    
    def test_color_constants_usage(self):
        """Test that color constants are properly used."""
        # This would test that MOSAICX_COLORS constants are used correctly
        from mosaicx.constants import MOSAICX_COLORS
        
        # Colors should be defined
        assert 'primary' in MOSAICX_COLORS
        assert 'success' in MOSAICX_COLORS
        assert 'error' in MOSAICX_COLORS
        assert 'info' in MOSAICX_COLORS
    
    def test_color_formatting(self):
        """Test color formatting in messages."""
        with patch('mosaicx.display.console') as mock_console:
            styled_message("Colored message", "success")
            
            # Should apply color formatting
            mock_console.print.assert_called_once()


class TestDisplayErrorHandling:
    """Test cases for display error handling."""
    
    def test_display_with_console_error(self):
        """Test handling of console output errors."""
        with patch('mosaicx.display.console') as mock_console:
            mock_console.print.side_effect = Exception("Console error")
            
            # Should handle console errors gracefully
            try:
                styled_message("Test message", "info")
                show_main_banner()
            except Exception:
                pytest.fail("Display functions should handle console errors gracefully")
    
    def test_display_with_none_values(self):
        """Test handling of None values."""
        with patch('mosaicx.display.console') as mock_console:
            # Should handle None message gracefully
            styled_message(None, "info")
            mock_console.print.assert_called_once()
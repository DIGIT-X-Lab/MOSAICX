"""Utilities for MOSAICX library."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def validate_date_format(date_string: str, expected_format: str = "YYYY-MM-DD") -> bool:
    """Validate date string format.
    
    Args:
        date_string: Date string to validate
        expected_format: Expected format pattern
        
    Returns:
        True if date is valid
    """
    try:
        if expected_format == "YYYY-MM-DD":
            datetime.strptime(date_string, "%Y-%m-%d")
        elif expected_format == "MM/DD/YYYY":
            datetime.strptime(date_string, "%m/%d/%Y")
        elif expected_format == "DD-MM-YYYY":
            datetime.strptime(date_string, "%d-%m-%Y")
        else:
            # Try common formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
                try:
                    datetime.strptime(date_string, fmt)
                    return True
                except ValueError:
                    continue
            return False
        return True
    except ValueError:
        return False


def validate_pattern(value: str, pattern: str) -> bool:
    """Validate value against regex pattern.
    
    Args:
        value: Value to validate
        pattern: Regex pattern
        
    Returns:
        True if value matches pattern
    """
    try:
        return bool(re.match(pattern, value))
    except re.error:
        return False


def generate_cache_key(content: str, config_dict: Dict[str, Any]) -> str:
    """Generate cache key from content and configuration.
    
    Args:
        content: Text content
        config_dict: Configuration dictionary
        
    Returns:
        Cache key string
    """
    content_hash = hashlib.md5(content.encode()).hexdigest()
    config_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
    return f"{content_hash}_{config_hash}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Trim underscores from ends
    sanitized = sanitized.strip('_')
    return sanitized


def format_confidence_score(score: float) -> str:
    """Format confidence score for display.
    
    Args:
        score: Confidence score (0-1)
        
    Returns:
        Formatted score string
    """
    percentage = score * 100
    if percentage >= 90:
        return f"[green]{percentage:.1f}%[/green]"
    elif percentage >= 70:
        return f"[yellow]{percentage:.1f}%[/yellow]"
    else:
        return f"[red]{percentage:.1f}%[/red]"


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough estimation: ~4 characters per token
    return len(text) // 4


def chunk_text(text: str, max_tokens: int = 4000) -> List[str]:
    """Split text into chunks that fit token limits.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    max_chars = max_tokens * 4  # Rough estimation
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
            else:
                # Single sentence is too long, split by words
                words = sentence.split()
                for word in words:
                    if len(current_chunk + word) <= max_chars:
                        current_chunk += word + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = word + " "
                        else:
                            # Single word is too long, just add it
                            chunks.append(word)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def merge_extraction_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple extraction results.
    
    Args:
        results: List of extraction result dictionaries
        
    Returns:
        Merged result dictionary
    """
    if not results:
        return {}
    
    if len(results) == 1:
        return results[0]
    
    merged = {}
    
    # Get all unique keys
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # Merge each field
    for key in all_keys:
        values = []
        for result in results:
            if key in result and result[key] is not None:
                values.append(result[key])
        
        if not values:
            merged[key] = None
        elif len(values) == 1:
            merged[key] = values[0]
        else:
            # For strings, take the longest non-empty value
            if all(isinstance(v, str) for v in values):
                non_empty = [v for v in values if v.strip()]
                if non_empty:
                    merged[key] = max(non_empty, key=len)
                else:
                    merged[key] = values[0]
            
            # For lists, merge and deduplicate
            elif all(isinstance(v, list) for v in values):
                merged_list = []
                for v in values:
                    for item in v:
                        if item not in merged_list:
                            merged_list.append(item)
                merged[key] = merged_list
            
            # For numbers, take the average
            elif all(isinstance(v, (int, float)) for v in values):
                merged[key] = sum(values) / len(values)
            
            # For other types, take the first value
            else:
                merged[key] = values[0]
    
    return merged


def find_pdf_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Find all PDF files in directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of PDF file paths
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    if recursive:
        return list(dir_path.rglob("*.pdf"))
    else:
        return list(dir_path.glob("*.pdf"))


def create_progress_callback(console=None):
    """Create a progress callback function.
    
    Args:
        console: Rich console for output
        
    Returns:
        Progress callback function
    """
    def progress_callback(current: int, total: int, current_file: str):
        if console:
            percentage = (current / total) * 100
            console.print(f"[{percentage:5.1f}%] Processing: {Path(current_file).name}")
        else:
            print(f"Processing {current}/{total}: {current_file}")
    
    return progress_callback


def format_processing_time(seconds: float) -> str:
    """Format processing time for display.
    
    Args:
        seconds: Processing time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """Extract common medical entities from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of entity types and found entities
    """
    entities = {
        "medications": [],
        "conditions": [],
        "procedures": [],
        "anatomical_sites": [],
        "measurements": []
    }
    
    # Simple pattern matching (could be enhanced with NLP models)
    medication_patterns = [
        r'\b[A-Z][a-z]+(?:in|ol|ide|ate|cin|ine)\b',
        r'\b(?:mg|mcg|ml|cc|units?)\b'
    ]
    
    for pattern in medication_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["medications"].extend(matches)
    
    # Measurements
    measurement_pattern = r'\b\d+(?:\.\d+)?\s*(?:mm|cm|kg|lbs?|mg|mcg|ml|cc)\b'
    measurements = re.findall(measurement_pattern, text, re.IGNORECASE)
    entities["measurements"].extend(measurements)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

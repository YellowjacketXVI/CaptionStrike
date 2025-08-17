"""
Token generation and management for CaptionStrike

Provides ULID-based unique identifiers for all processed media files.
ULIDs are lexicographically sortable and contain timestamp information.
"""

from ulid import ULID
from typing import Optional
import re


def generate_token() -> str:
    """Generate a new ULID token with TKN- prefix.
    
    Returns:
        str: Token in format "TKN-<ULID>"
    """
    return f"TKN-{ULID()}"


def extract_token_from_filename(filename: str) -> Optional[str]:
    """Extract token from a filename if present.
    
    Args:
        filename: Filename that may contain a token
        
    Returns:
        str or None: The token if found, None otherwise
    """
    # Look for pattern __TKN-<ULID> in filename
    match = re.search(r'__TKN-([0-9A-HJKMNP-TV-Z]{26})', filename.upper())
    if match:
        return f"TKN-{match.group(1)}"
    return None


def extract_token_from_caption(caption: str) -> Optional[str]:
    """Extract token from caption text if present.
    
    Args:
        caption: Caption text that may end with [TKN-<ULID>]
        
    Returns:
        str or None: The token if found, None otherwise
    """
    # Look for pattern [TKN-<ULID>] at end of caption
    match = re.search(r'\[TKN-([0-9A-HJKMNP-TV-Z]{26})\]$', caption.upper())
    if match:
        return f"TKN-{match.group(1)}"
    return None


def add_token_to_filename(base_name: str, token: str) -> str:
    """Add token to a base filename.
    
    Args:
        base_name: Base filename without extension
        token: Token to add (should include TKN- prefix)
        
    Returns:
        str: Filename with token added
    """
    return f"{base_name}__{token}"


def add_token_to_caption(caption: str, token: str) -> str:
    """Add token to caption text.
    
    Args:
        caption: Caption text
        token: Token to add (should include TKN- prefix)
        
    Returns:
        str: Caption with token appended in brackets
    """
    # Remove any existing token first
    caption = re.sub(r'\s*\[TKN-[0-9A-HJKMNP-TV-Z]{26}\]$', '', caption, flags=re.IGNORECASE)
    return f"{caption.strip()} [{token}]"


def is_valid_token(token: str) -> bool:
    """Check if a token is valid ULID format.
    
    Args:
        token: Token to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not token.startswith('TKN-'):
        return False
    
    ulid_part = token[4:]  # Remove TKN- prefix
    return len(ulid_part) == 26 and re.match(r'^[0-9A-HJKMNP-TV-Z]{26}$', ulid_part) is not None


def safe_filename(name: str) -> str:
    """Convert a string to a safe filename.
    
    Args:
        name: Original name
        
    Returns:
        str: Safe filename with problematic characters replaced
    """
    # Replace spaces and problematic characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    safe = re.sub(r'\s+', '_', safe)
    safe = re.sub(r'_{2,}', '_', safe)  # Replace multiple underscores with single
    return safe.strip('_')

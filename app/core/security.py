import secrets
import hashlib
from typing import Optional
from datetime import datetime, timedelta

def generate_api_key() -> str:
    """
    Generate a secure random API key
    
    Returns:
        str: 32-character hexadecimal API key
    """
    return secrets.token_hex(32)


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage
    
    Args:
        api_key: Raw API key
    
    Returns:
        str: SHA256 hash of the API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(raw_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against its hash
    
    Args:
        raw_key: Raw API key to verify
        hashed_key: Stored hash
    
    Returns:
        bool: True if keys match
    """
    return hash_api_key(raw_key) == hashed_key


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize a filename for safe storage
    
    Args:
        filename: Original filename
        max_length: Maximum length (default: 100)
    
    Returns:
        str: Sanitized filename
    """
    # Remove path components
    filename = filename.split("/")[-1].split("\\")[-1]
    
    # Replace unsafe characters
    unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Truncate if too long
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = max_length - len(ext) - 1
        filename = f"{name[:max_name_length]}.{ext}" if ext else name[:max_length]
    
    return filename


def generate_verification_code(length: int = 6) -> str:
    """
    Generate a numeric verification code
    
    Args:
        length: Code length (default: 6)
    
    Returns:
        str: Numeric verification code
    """
    return ''.join([str(secrets.randbelow(10)) for _ in range(length)])


def is_expired(created_at: datetime, expiry_minutes: int = 60) -> bool:
    """
    Check if a timestamp has expired
    
    Args:
        created_at: Creation timestamp
        expiry_minutes: Expiry time in minutes (default: 60)
    
    Returns:
        bool: True if expired
    """
    expiry_time = created_at + timedelta(minutes=expiry_minutes)
    return datetime.utcnow() > expiry_time


def mask_email(email: str) -> str:
    """
    Mask an email address for privacy
    
    Args:
        email: Email address to mask
    
    Returns:
        str: Masked email (e.g., "j***@example.com")
    """
    if '@' not in email:
        return email
    
    local, domain = email.split('@')
    
    if len(local) <= 2:
        masked_local = local[0] + '*'
    else:
        masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
    
    return f"{masked_local}@{domain}"


def validate_image_extension(filename: str) -> bool:
    """
    Validate if filename has an allowed image extension
    
    Args:
        filename: Filename to validate
    
    Returns:
        bool: True if valid image extension
    """
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    ext = '.' + filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    return ext in allowed_extensions


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token
    
    Args:
        length: Token length in bytes (default: 32)
    
    Returns:
        str: URL-safe random token
    """
    return secrets.token_urlsafe(length)


class RateLimiter:
    """
    Simple in-memory rate limiter
    (For production, use Redis-based rate limiting)
    """
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(
        self,
        key: str,
        max_requests: int = 10,
        window_seconds: int = 60
    ) -> bool:
        """
        Check if a request is allowed under rate limit
        
        Args:
            key: Unique identifier (e.g., IP address, user ID)
            max_requests: Maximum requests allowed (default: 10)
            window_seconds: Time window in seconds (default: 60)
        
        Returns:
            bool: True if request is allowed
        """
        now = datetime.utcnow()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove expired requests
        cutoff = now - timedelta(seconds=window_seconds)
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > cutoff
        ]
        
        # Check limit
        if len(self.requests[key]) >= max_requests:
            return False
        
        # Add new request
        self.requests[key].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()
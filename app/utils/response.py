from typing import Any, Optional
from datetime import datetime

def success_response(
    data: Any = None,
    message: str = "Success",
    status_code: int = 200
) -> dict:
    """
    Standard success response format
    
    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code (default: 200)
    
    Returns:
        dict: {
            "success": true,
            "message": "...",
            "data": {...},
            "timestamp": "2024-01-20T12:00:00"
        }
    """
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }

def error_response(
    message: str = "Error",
    errors: Optional[dict] = None,
    status_code: int = 400
) -> dict:
    """
    Standard error response format
    
    Args:
        message: Error message
        errors: Detailed error information (optional)
        status_code: HTTP status code (default: 400)
    
    Returns:
        dict: {
            "success": false,
            "message": "...",
            "errors": {...},
            "timestamp": "2024-01-20T12:00:00"
        }
    """
    response = {
        "success": False,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if errors:
        response["errors"] = errors
    
    return response

def paginated_response(
    items: list,
    total: int,
    page: int = 1,
    page_size: int = 10,
    message: str = "Success"
) -> dict:
    """
    Standard paginated response format
    
    Args:
        items: List of items for current page
        total: Total number of items
        page: Current page number
        page_size: Number of items per page
        message: Success message
    
    Returns:
        dict: {
            "success": true,
            "message": "...",
            "data": {
                "items": [...],
                "pagination": {
                    "total": 100,
                    "page": 1,
                    "page_size": 10,
                    "total_pages": 10
                }
            },
            "timestamp": "2024-01-20T12:00:00"
        }
    """
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "success": True,
        "message": message,
        "data": {
            "items": items,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }
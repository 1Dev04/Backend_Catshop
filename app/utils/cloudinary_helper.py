import cloudinary
import cloudinary.uploader
import cloudinary.api
from fastapi import UploadFile, HTTPException
import os
from datetime import datetime
from app.core.config import settings

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True
)

async def upload_image_to_cloudinary(
    file: UploadFile,
    user_id: str,
    folder: str = "cat_shop"
) -> dict:
    """
    Upload image to Cloudinary
    
    Args:
        file: UploadFile from FastAPI
        user_id: Firebase user ID
        folder: Cloudinary folder name (default: "cat_shop")
    
    Returns:
        dict: {
            "url": "https://res.cloudinary.com/...",
            "secure_url": "https://res.cloudinary.com/...",
            "public_id": "cat_shop/user123/image456",
            "width": 1920,
            "height": 1080,
            "format": "jpg",
            "resource_type": "image",
            "created_at": "2024-01-20T12:00:00",
            "bytes": 123456,
            "thumbnail_url": "https://res.cloudinary.com/.../c_thumb,w_200,h_200/..."
        }
    
    Raises:
        HTTPException: If upload fails
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()  # Get position (file size)
        file.file.seek(0)  # Reset to beginning
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: 10MB"
            )
        
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_{timestamp}"
        
        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            file.file,
            folder=f"{folder}/{user_id}",
            public_id=filename,
            overwrite=True,
            resource_type="image",
            # Transformations
            eager=[
                {
                    "width": 1920,
                    "height": 1080,
                    "crop": "limit",
                    "quality": "auto",
                    "fetch_format": "auto"
                },
                {
                    "width": 200,
                    "height": 200,
                    "crop": "thumb",
                    "gravity": "auto",
                    "quality": "auto"
                }
            ],
            eager_async=False,
        )
        
        # Generate thumbnail URL
        thumbnail_url = cloudinary.CloudinaryImage(result['public_id']).build_url(
            width=200,
            height=200,
            crop="thumb",
            gravity="auto",
            quality="auto"
        )
        
        return {
            "url": result.get("url"),
            "secure_url": result.get("secure_url"),
            "public_id": result.get("public_id"),
            "width": result.get("width"),
            "height": result.get("height"),
            "format": result.get("format"),
            "resource_type": result.get("resource_type"),
            "created_at": result.get("created_at"),
            "bytes": result.get("bytes"),
            "thumbnail_url": thumbnail_url
        }
    
    except cloudinary.exceptions.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cloudinary upload error: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


async def delete_image_from_cloudinary(public_id: str) -> dict:
    """
    Delete image from Cloudinary
    
    Args:
        public_id: Cloudinary public ID (e.g., "cat_shop/user123/image456")
    
    Returns:
        dict: {"result": "ok"} or {"result": "not found"}
    
    Raises:
        HTTPException: If deletion fails
    """
    try:
        result = cloudinary.uploader.destroy(public_id)
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete image: {str(e)}"
        )


def get_cloudinary_url(
    public_id: str,
    width: int = None,
    height: int = None,
    crop: str = "limit",
    quality: str = "auto"
) -> str:
    """
    Generate Cloudinary URL with transformations
    
    Args:
        public_id: Cloudinary public ID
        width: Image width (optional)
        height: Image height (optional)
        crop: Crop mode (default: "limit")
        quality: Image quality (default: "auto")
    
    Returns:
        str: Transformed image URL
    """
    transformation = {}
    
    if width:
        transformation["width"] = width
    if height:
        transformation["height"] = height
    if crop:
        transformation["crop"] = crop
    if quality:
        transformation["quality"] = quality
    
    return cloudinary.CloudinaryImage(public_id).build_url(**transformation)
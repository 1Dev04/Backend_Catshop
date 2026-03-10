from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class ClothingResponse(BaseModel):
    """
    Response model for clothing items
    ใช้สำหรับส่งข้อมูลไปยัง Flutter
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    image_url: str
    clothing_name: str
    description: str
    price: str


class ClothingDetailResponse(ClothingResponse):
    """
    Detailed response model with timestamps
    """
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ClothingCreate(BaseModel):
    """
    Model for creating new clothing item
    """
    image_url: str = Field(..., description="URL of clothing image")
    clothing_name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    price: float = Field(..., gt=0, description="Price must be greater than 0")
    display_order: int = Field(default=0)
    is_active: bool = Field(default=True)


class ClothingUpdate(BaseModel):
    """
    Model for updating clothing item
    """
    image_url: Optional[str] = None
    clothing_name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    display_order: Optional[int] = None
    is_active: Optional[bool] = None
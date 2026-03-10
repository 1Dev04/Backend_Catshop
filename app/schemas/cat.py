from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List
from datetime import datetime

# ============================================
# CREATE SCHEMA
# ============================================
class CatCreate(BaseModel):
    """Schema สำหรับสร้างข้อมูลแมวใหม่"""
    
    # ข้อมูลพื้นฐาน
    cat_color: Optional[str] = Field(None, description="สีของแมว เช่น orange, black+white")
    breed: Optional[str] = Field(None, description="สายพันธุ์")
    age: Optional[int] = Field(None, description="อายุ (เดือน)")
    age_category: Optional[str] = Field("adult", description="kitten/young/adult/senior")
    gender: Optional[int] = Field(0, description="0=ไม่ระบุ, 1=ผู้, 2=เมีย")
    
    # น้ำหนักและสภาพร่างกาย
    weight: Optional[float] = Field(None, description="น้ำหนัก (kg)")
    body_condition_score: Optional[int] = Field(None, ge=1, le=9, description="BCS 1-9")
    body_condition: Optional[str] = Field(None, description="underweight/lean/ideal/overweight/obese")
    body_condition_description: Optional[str] = Field(None, description="คำอธิบายสภาพร่างกาย")
    bmi: Optional[float] = Field(None, description="ดัชนีมวลกาย")
    
    # ขนาดส่วนต่างๆ
    chest_cm: Optional[float] = Field(None, description="รอบอก (cm)")
    neck_cm: Optional[float] = Field(None, description="รอบคอ (cm)")
    waist_cm: Optional[float] = Field(None, description="รอบเอว (cm)")
    body_length_cm: Optional[float] = Field(None, description="ความยาวลำตัว (cm)")
    back_length_cm: Optional[float] = Field(None, description="ความยาวหลัง (cm)")
    leg_length_cm: Optional[float] = Field(None, description="ความยาวขา (cm)")
    
    # ขนาดเสื้อผ้า
    size_category: Optional[str] = Field(None, description="XS/S/M/L/XL")
    size_recommendation: Optional[str] = Field(None, description="คำแนะนำขนาด")
    size_ranges: Optional[Dict] = Field(None, description="ช่วงขนาด")
    
    # ข้อมูลการวิเคราะห์
    posture: Optional[str] = Field(None, description="ท่าทาง")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="ความมั่นใจ")
    quality_flag: Optional[str] = Field(None, description="คุณภาพภาพ")
    bounding_box: Optional[List[float]] = Field(None, description="พิกัดแมวในภาพ")
    
    # ข้อมูลรูปภาพ
    image_url: Optional[str] = Field(None, description="URL รูปภาพ")
    thumbnail_url: Optional[str] = Field(None, description="URL thumbnail")
    
    # Metadata
    analysis_version: Optional[str] = Field("5.0", description="เวอร์ชัน algorithm")
    analysis_method: Optional[str] = Field("cv_heuristic_v5_professional", description="วิธีการวิเคราะห์")

# ============================================
# UPDATE SCHEMA
# ============================================
class CatUpdate(BaseModel):
    """Schema สำหรับอัปเดตข้อมูลแมว (ทุกฟิลด์ optional)"""
    
    cat_color: Optional[str] = None
    breed: Optional[str] = None
    age: Optional[int] = None
    age_category: Optional[str] = None
    gender: Optional[int] = None
    weight: Optional[float] = None
    body_condition_score: Optional[int] = Field(None, ge=1, le=9)
    body_condition: Optional[str] = None
    body_condition_description: Optional[str] = None
    bmi: Optional[float] = None
    chest_cm: Optional[float] = None
    neck_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    body_length_cm: Optional[float] = None
    back_length_cm: Optional[float] = None
    leg_length_cm: Optional[float] = None
    size_category: Optional[str] = None
    size_recommendation: Optional[str] = None
    size_ranges: Optional[Dict] = None
    posture: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    quality_flag: Optional[str] = None
    bounding_box: Optional[List[float]] = None
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

# ============================================
# RESPONSE SCHEMA
# ============================================
class CatResponse(BaseModel):
    """Schema สำหรับ Response"""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    firebase_uid: Optional[str]
    cat_color: Optional[str]
    breed: Optional[str]
    age: Optional[int]
    age_category: Optional[str]
    gender: Optional[int]
    weight: Optional[float]
    body_condition_score: Optional[int]
    body_condition: Optional[str]
    body_condition_description: Optional[str]
    bmi: Optional[float]
    chest_cm: Optional[float]
    neck_cm: Optional[float]
    waist_cm: Optional[float]
    body_length_cm: Optional[float]
    back_length_cm: Optional[float]
    leg_length_cm: Optional[float]
    size_category: Optional[str]
    size_recommendation: Optional[str]
    size_ranges: Optional[Dict]
    posture: Optional[str]
    confidence: Optional[float]
    quality_flag: Optional[str]
    bounding_box: Optional[List]
    image_url: Optional[str]
    thumbnail_url: Optional[str]
    analysis_version: Optional[str]
    analysis_method: Optional[str]
    detected_at: Optional[datetime]
    updated_at: Optional[datetime]

# ============================================
# ANALYSIS RESULT SCHEMA (สำหรับ endpoint /analysis/save)
# ============================================
class AnalysisResultSchema(BaseModel):
    """Schema สำหรับผลการวิเคราะห์จาก CatAnalyzer"""
    
    firebase_uid: str
    cat_color: str
    breed: str
    age_category: str
    weight_kg: float
    body_condition_score: int
    body_condition: str
    body_condition_description: str
    bmi: float
    measurements: Dict[str, float]  # chest_cm, neck_cm, waist_cm, etc.
    size_category: str
    size_ranges: Dict
    size_recommendation: str
    posture: str
    confidence: float
    quality_flag: str
    bounding_box: Optional[List[float]] = None
    image_path: str
    analysis_version: str
    analysis_method: str
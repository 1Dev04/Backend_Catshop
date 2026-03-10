from sqlalchemy import Column, Integer, String, Numeric, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.db.database import Base

class Cat(Base):
    __tablename__ = "cat"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # ข้อมูลพื้นฐาน
    cat_color = Column(String(100), nullable=True, comment='สีของแมว (เช่น orange+white)')
    breed = Column(String(100), nullable=True, comment='สายพันธุ์')
    age = Column(Integer, nullable=True, comment='อายุ (เดือน)')
    age_category = Column(String(20), default='adult', comment='ช่วงอายุ: kitten/young/adult/senior')
    gender = Column(Integer, default=0, comment='0=ไม่ระบุ, 1=ผู้, 2=เมีย')
    
    # น้ำหนักและสภาพร่างกาย
    weight = Column(Numeric(5, 2), nullable=True, comment='น้ำหนัก (kg)')
    body_condition_score = Column(Integer, nullable=True, comment='BCS 1-9 คะแนน')
    body_condition = Column(String(30), nullable=True, comment='underweight/lean/ideal/overweight/obese')
    body_condition_description = Column(Text, nullable=True, comment='คำแนะนำเกี่ยวกับสภาพร่างกาย')
    bmi = Column(Numeric(5, 2), nullable=True, comment='ดัชนีมวลกายแมว')
    
    # ขนาดส่วนต่างๆ
    chest_cm = Column(Numeric(6, 2), nullable=True, comment='รอบอก (cm)')
    neck_cm = Column(Numeric(6, 2), nullable=True, comment='รอบคอ (cm)')
    waist_cm = Column(Numeric(6, 2), nullable=True, comment='รอบเอว (cm)')
    body_length_cm = Column(Numeric(6, 2), nullable=True, comment='ความยาวลำตัว (cm)')
    back_length_cm = Column(Numeric(6, 2), nullable=True, comment='ความยาวหลัง (cm)')
    leg_length_cm = Column(Numeric(6, 2), nullable=True, comment='ความยาวขา (cm)')
    
    # ขนาดเสื้อผ้า
    size_category = Column(String(5), nullable=True, comment='XS/S/M/L/XL')
    size_recommendation = Column(Text, nullable=True, comment='คำแนะนำขนาด')
    size_ranges = Column(JSON, nullable=True, comment='ช่วงขนาด JSON')
    
    # ข้อมูลการวิเคราะห์
    posture = Column(String(20), nullable=True, comment='ท่าทาง: lying/sitting/standing/curled')
    confidence = Column(Numeric(5, 4), nullable=True, comment='ความมั่นใจ 0-1')
    quality_flag = Column(String(20), nullable=True, comment='คุณภาพภาพ: excellent/good/medium/poor')
    bounding_box = Column(JSON, nullable=True, comment='พิกัดแมวในภาพ [x1, y1, x2, y2]')
    
    # ข้อมูลรูปภาพ
    image_url = Column(Text, nullable=True, comment='URL รูปภาพต้นฉบับ')
    thumbnail_url = Column(Text, nullable=True, comment='URL รูปภาพขนาดย่อ')
    
    # ข้อมูลเจ้าของ
    firebase_uid = Column(String(128), nullable=True, index=True, comment='Firebase UID ของเจ้าของ')
    
    # Metadata
    analysis_version = Column(String(10), default='5.0', comment='เวอร์ชัน algorithm')
    analysis_method = Column(String(50), default='cv_heuristic_v5_professional', comment='วิธีการวิเคราะห์')
    detected_at = Column(DateTime(timezone=True), server_default=func.now(), comment='วันที่ตรวจจับ')
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment='วันที่อัปเดต')
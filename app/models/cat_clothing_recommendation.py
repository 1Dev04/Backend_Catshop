from sqlalchemy import Column, Integer, Numeric, Text, Boolean, TIMESTAMP, ForeignKey, CheckConstraint, Index, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from catshop_system.backend_catshop.app.db.database import Base

class CatClothingRecommendation(Base):
    """
    Model สำหรับเก็บคำแนะนำเสื้อผ้าที่เหมาะกับแมว
    """
    __tablename__ = "cat_clothing_recommendations"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign Keys
    cat_id = Column(
        Integer,
        ForeignKey('cats.id', ondelete='CASCADE'),
        nullable=False,
        comment="cats.id"
    )
    clothing_id = Column(
        Integer,
        ForeignKey('cat_clothing.id', ondelete='CASCADE'),
        nullable=False,
        comment="cat_clothing.id"
    )
    
    # Matching Logic
    match_score = Column(
        Numeric(4, 3),
        CheckConstraint('match_score BETWEEN 0 AND 1'),
        nullable=False,
        comment="คะแนนความเหมาะสม (0-1)"
    )
    match_name = Column(String(100), nullable=True, comment="ชื่อที่ตรงกัน")
    match_size = Column(Boolean, default=False, comment="ขนาดตรงกันไหม")
    match_weight = Column(Boolean, default=False, comment="น้ำหนักตรงกันไหม")
    match_chest = Column(Boolean, default=False, comment="รอบอกตรงกันไหม")
    match_neck = Column(Boolean, default=False, comment="รอบคอตรงกันไหม")
    
    discount_price = Column(Numeric(10, 2), nullable=True, comment="ราคาพิเศษ")
    reason = Column(Text, nullable=True, comment="เหตุผลที่แนะนำ")
    
    # User Interaction
    is_selected = Column(Boolean, default=False, comment="User เลือกสินค้านี้หรือไม่")
    
    # Timestamp
    recommended_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        comment="วันที่แนะนำ"
    )
    
    # Relationships
    cat = relationship("Cat", backref="recommendations")
    clothing = relationship("CatClothing", backref="recommendations")
    
    # Constraints & Indexes
    __table_args__ = (
        UniqueConstraint('cat_id', 'clothing_id', name='uq_cat_clothing'),
        Index('idx_reco_cat_id', 'cat_id'),
        Index('idx_reco_score', 'match_score'),
    )
    
    def __repr__(self):
        return f"<Recommendation(cat_id={self.cat_id}, clothing_id={self.clothing_id}, score={self.match_score})>"
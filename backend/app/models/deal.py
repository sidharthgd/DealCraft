from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship, column_property
from sqlalchemy import func, select
from app.core.database import Base


class Deal(Base):
    __tablename__ = "deals"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="deals")
    documents = relationship("Document", back_populates="deal", cascade="all, delete-orphan")
    memos = relationship("Memo", back_populates="deal", cascade="all, delete-orphan")

    # Computed property for document count
    @property
    def document_count(self):
        return len(self.documents) 
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum
# Removed import: from .tag import document_tag_association  # Temporarily removed


class DocumentStatus(enum.Enum):
    uploading = "uploading"
    processing = "processing"
    completed = "completed"
    error = "error"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)  # Legacy local file path
    storage_path = Column(String, nullable=True)  # Cloud storage path (GCS)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String, nullable=False)
    deal_id = Column(String, ForeignKey("deals.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.uploading)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    category = Column(String, nullable=True)  # Added for document categorization

    # Relationships
    deal = relationship("Deal", back_populates="documents")
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    # Temporarily completely removed tags relationship
    # tags = relationship(
    #     "DocumentTag",
    #     secondary=document_tag_association,
    #     back_populates="documents",
    #     lazy='select'  # Use select loading to avoid session issues
    # )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks") 
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Table, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

# Association table between documents and tags

document_tag_association = Table(
    "document_tag_association",
    Base.metadata,
    Column("document_id", String, ForeignKey("documents.id"), primary_key=True),
    Column("tag_id", String, ForeignKey("tags.id"), primary_key=True),
)


class DocumentTag(Base):
    """Simple tag entity that can be attached to documents for quick filtering/searching."""

    __tablename__ = "tags"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back-ref will be declared on Document
    documents = relationship(
        "Document",
        secondary=document_tag_association,
        # Temporarily removed back_populates to fix session issues
        # back_populates="tags",
    ) 
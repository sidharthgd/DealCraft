from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from app.models.document import DocumentStatus


class DocumentBase(BaseModel):
    name: str
    file_path: str
    file_size: int
    content_type: str


class DocumentCreate(DocumentBase):
    deal_id: str
    user_id: str


class DocumentUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[DocumentStatus] = None


class Document(DocumentBase):
    id: str
    deal_id: str
    user_id: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentChunkBase(BaseModel):
    content: str
    chunk_index: int
    start_char: int
    end_char: int


class DocumentChunkCreate(DocumentChunkBase):
    document_id: str


class DocumentChunk(DocumentChunkBase):
    id: str
    document_id: str
    created_at: datetime

    class Config:
        from_attributes = True 
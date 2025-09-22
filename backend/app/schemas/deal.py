from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class DealBase(BaseModel):
    name: str
    description: Optional[str] = None


class DealCreate(DealBase):
    pass


class DealUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class Deal(DealBase):
    id: str
    user_id: str
    document_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 
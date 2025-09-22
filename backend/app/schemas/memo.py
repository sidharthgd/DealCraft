from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MemoBase(BaseModel):
    title: str
    content: str


class MemoCreate(MemoBase):
    deal_id: str
    user_id: str


class MemoUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None


class Memo(MemoBase):
    id: str
    deal_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

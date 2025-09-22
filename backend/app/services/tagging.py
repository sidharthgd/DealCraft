import re
import uuid
from typing import List
from app.models.tag import DocumentTag
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


class AutoTaggingService:
    """A naive keyword-based tagger; can be swapped for an ML model later."""

    _rules = {
        r"\bincome\b.*\bstatement\b": "Income Statement",
        r"\bprofit\b.*\bloss\b": "Income Statement", 
        r"\bp\s*&\s*l\b": "Income Statement",
        r"\bbalance\b.*\bsheet\b": "Balance Sheet",
        r"\bcash\b.*\bflow\b": "Cash Flow",
        r"\bloi\b": "LOI",
        r"\bletter\b.*\bintent\b": "LOI",
        r"\bnda\b": "LOI",
        r"\bcim\b": "CIM",
        r"\bconfidential\b.*\bmemorandum\b": "CIM",
        r"\bdue\b.*\bdiligence\b": "Diligence Tracker",
        r"\bdiligence\b.*\btracker\b": "Diligence Tracker",
        r"\bchecklist\b": "Diligence Tracker",
        r"\bcustomer\b.*\blist\b": "Customer List",
        r"\bcustomer\b.*\bcontracts\b": "Customer List",
    }

    async def suggest_tags(self, filename: str, session: AsyncSession) -> List[DocumentTag]:
        lower = filename.lower()
        tags: List[DocumentTag] = []
        for pattern, tag_name in self._rules.items():
            if re.search(pattern, lower):
                # Fetch or create tag
                result = await session.execute(
                    select(DocumentTag).where(DocumentTag.name == tag_name)
                )
                tag: DocumentTag | None = result.scalar_one_or_none()
                if not tag:
                    tag = DocumentTag(id=str(uuid.uuid4()), name=tag_name)
                    session.add(tag)
                tags.append(tag)
        return tags


auto_tagging_service = AutoTaggingService() 